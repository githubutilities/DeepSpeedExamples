# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import sys
import os
import random

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from deepspeed.runtime.utils import see_memory_usage

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from model.model_utils import create_hf_model
from utils_inference import load_model_on_gpus

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        '-m', "--model_name_or_path_baseline",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--serving_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--start_flask",
        action='store_true',
        default=False,
        required=False,
    )
    parser.add_argument(
        "--use_reward",
        action='store_true',
        default=False,
        required=False,
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        help="Path to pretrained model",
        required=False,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_gpu",
        type=int,
        default=8,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="Chinese",
                        choices=["English", "Chinese", "Japanese"])
    parser.add_argument("--device",
        type=str,
        default="cuda")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
 
    args = parser.parse_args()

    return args


def reward_inference(model,
             tokenizer,
             inputs,
             **kwargs):
    result = model(inputs.input_ids, attention_mask=inputs.attention_mask, return_dict=True)
    return result.rewards.cpu().detach().numpy().tolist()


def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
             num_return_sequences=1,
             max_new_tokens=100):

    if hasattr(model, 'backbone_model'):
        model = model.backbone_model
    generate_ids = model.generate(inputs.input_ids,
                                  attention_mask=inputs.attention_mask,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    generate_ids = torch.where(generate_ids == -1, 0, generate_ids)
    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def generate_constrastive_search(model,
                                 tokenizer,
                                 inputs,
                                 top_k=4,
                                 penalty_alpha=0.6,
                                 num_return_sequences=1,
                                 max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  attention_mask=inputs.attention_mask,
                                  top_k=top_k,
                                  penalty_alpha=penalty_alpha,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def print_utils(gen_output):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        for i in range(len(gen_output)):
            print()
            print(gen_output[i])
            print()

def prompt_chatglm(args, model, tokenizer, device, prompt):
    max_length = args.max_new_tokens
    top_p = 0.6
    temperature = 0.95
    history = []
    for response, history in model.stream_chat(tokenizer, prompt, history, max_length=max_length, top_p=top_p, temperature=temperature):
        print(prompt, response, history)

import json
import logging
import traceback
logging.basicConfig(level=logging.INFO,filename='log.txt',datefmt='%Y/%m/%d %H:%M:%S',\
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

from flask import Flask, request
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def infer_loop(prompt=None):
    #local_rank = int(os.getenv('LOCAL_RANK', '0'))
    global global_dict
    model = global_dict['model']
    tokenizer = global_dict['tokenizer']
    device = torch.device('cuda')
    args = global_dict['args']
    while True:
        if args.num_gpu > 1:
            torch.distributed.barrier()
        tensor_params = {'dtype': torch.int64, 'device': device}
        if prompt is not None:
            tok_params = {
                'padding': 'longest',
            }
            inputs = tokenizer(prompt, return_tensors="pt", **tok_params).to(device)
            input_ids = inputs.input_ids
            shape_tensor = torch.tensor(input_ids.shape, **tensor_params)
            if args.num_gpu > 1:
                print('a send', shape_tensor)
                torch.distributed.broadcast(tensor=shape_tensor, src=0)
                print('b send', input_ids)
                torch.distributed.broadcast(tensor=input_ids, src=0)
        else:
            input_ids = None
            shape_tensor = torch.zeros((2,), **tensor_params)
            if args.num_gpu > 1:
                torch.distributed.broadcast(tensor=shape_tensor, src=0)
                print('a recv', shape_tensor)
                input_ids = torch.empty(shape_tensor.cpu().tolist(), **tensor_params)
                torch.distributed.broadcast(tensor=input_ids, src=0)
                print('b recv', input_ids)
        inputs = AttrDict()
        attention_mask = (input_ids != tokenizer.pad_token_id).type(input_ids.dtype)
        print('attention_mask', attention_mask)
        inputs.update({'input_ids': input_ids, 'attention_mask': attention_mask})
        if args.use_reward:
            fn = reward_inference
        else:
            fn = generate
        r_finetune_b = fn(model=model,
                            tokenizer=tokenizer, 
                            inputs=inputs,
                            num_beams=args.num_beams,
                            num_return_sequences=args.num_return_sequences,
                            max_new_tokens=args.max_new_tokens)
        print(r_finetune_b)
        if prompt is not None:
            break
    #torch.distributed.barrier()
    return r_finetune_b
 
global_dict = {}
@app.route("/generate/", methods=['POST'])
def api():
    if request.method == 'POST':
        global global_dict
        model = global_dict['model']
        tokenizer = global_dict['tokenizer']
        args = global_dict['args']
        device = torch.device('cuda')
        try:
            data = request.get_data().decode('utf8')
            data = json.loads(data)
            #prompts = data['query']
            prompts = []
            ret = []
            for e in data.get('query_list', []):
                e['history'] = e.get('history', []) + [e.get('query', '')]
                ret.append( {'history': e['history']} )
                cur = '\n'.join( e['history'] )
                prompts.append(cur)
            print('prompts', prompts)
            r_finetune_b = infer_loop(prompt=prompts)
            for idx, ans in enumerate(r_finetune_b):
                idx = idx // args.num_return_sequences
                if len(ret) <= idx:
                    print('warning generate index overflow')
                else:
                    if args.num_return_sequences > 1:
                        ret[idx]['answer'] = ret[idx].get('answer', [])
                        ret[idx]['answer'].append(ans)
                    else:
                        ret[idx]['answer'] = ans
            result = json.dumps({'ret_code': 0, 'result': ret}, ensure_ascii=False, indent=2)
        except:
            error = traceback.format_exc()
            result = json.dumps({'ret_code': -1, 'result': 'error', 'errmsg': error}, ensure_ascii=False)
            print(error)
            print(result)
            logger.info('request : {} \n response : {}'.format(data,result))
        return result
 
def prompt_eval(args, model_fintuned, tokenizer, device, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print("==========finetune: Greedy=========")
    r_finetune_g = generate(model_fintuned,
                            tokenizer,
                            inputs,
                            num_beams=1,
                            num_return_sequences=args.num_return_sequences,
                            max_new_tokens=args.max_new_tokens)
    print_utils(r_finetune_g)
    see_memory_usage("after init", True)
    # Note: we use the above simplest greedy search as the baseline. Users can also use other baseline methods,
    # such as beam search, multinomial sampling, and beam-search multinomial sampling.
    # We provide examples as below for users to try.

    print("==========finetune: Multinomial sampling=========")
    r_finetune_m = generate(model_fintuned, tokenizer, inputs,
                            num_beams=1,
                            do_sample=True,
                            num_return_sequences=args.num_return_sequences,
                            max_new_tokens=args.max_new_tokens)
    print_utils(r_finetune_m)
    print("==========finetune: Beam Search=========")
    r_finetune_b = generate(model_fintuned, tokenizer, inputs,
                            num_beams=args.num_beams,
                            num_return_sequences=args.num_return_sequences,
                            max_new_tokens=args.max_new_tokens)
    print_utils(r_finetune_b)
    print("==========finetune: Beam-search multinomial sampling=========")
    r_finetune_s = generate(model_fintuned, tokenizer, inputs,
                            num_beams=args.num_beams,
                            do_sample=True,
                            num_return_sequences=args.num_return_sequences,
                            max_new_tokens=args.max_new_tokens)
    print_utils(r_finetune_s)
    print("==========finetune: Diverse Beam Search=========")
    r_finetune_d = generate(model_fintuned, tokenizer, inputs,
                            num_beams=args.num_beams,
                            num_beam_groups=args.num_beam_groups,
                            num_return_sequences=args.num_return_sequences,
                            max_new_tokens=args.max_new_tokens)
    print_utils(r_finetune_d)
    print("==========finetune: Constrastive Search=========")
    r_finetune_c = generate_constrastive_search(model_fintuned, tokenizer, inputs,
                                                top_k=args.top_k,
                                                penalty_alpha=args.penalty_alpha,
                                                num_return_sequences=args.num_return_sequences,
                                                max_new_tokens=args.max_new_tokens)
    print_utils(r_finetune_c)
    print("====================prompt end=============================")
    print()
    print()


def main():
    args = parse_args()
    global global_dict
    print('global dict', len(global_dict))
    if len(global_dict) > 0:
        return
    global_dict['args'] = args
    if args.num_gpu == 1:
        os.environ['LOCAL_RANK'] = '0'
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    print(local_rank, world_size)
    if world_size > 1:
        torch.cuda.set_device(local_rank)

    device_name = args.device
    device = torch.device(device_name)
    model_class  = AutoModelForCausalLM
    tokenizer_class = AutoTokenizer
    config_class = AutoConfig
    model_dir = args.model_name_or_path_baseline
    tok_params = {
        'padding_side': 'left',
        'trust_remote_code': True,
    }
    if 'chatglm' in model_dir:
        print('chatglm tokenizer')
        from chatglm_6b.configuration_chatglm import ChatGLMConfig
        from chatglm_6b.tokenization_chatglm import ChatGLMTokenizer
        from chatglm_6b.modeling_chatglm import ChatGLMForConditionalGeneration
        config_class = ChatGLMConfig
        tokenizer_class = ChatGLMTokenizer
        model_class = ChatGLMForConditionalGeneration
        tokenizer = tokenizer_class.from_pretrained('chatglm_6b', fast_tokenizer=True, **tok_params)
    else:
        tokenizer = tokenizer_class.from_pretrained(
            model_dir, fast_tokenizer=True, **tok_params)

    print(args)
    if device_name == 'cpu':
        print('start loading')
        model = model_class.from_pretrained(
            model_dir, 
            offload_state_dict=True,
            #torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        #model = create_hf_model(model_class, model_dir, tokenizer, None)
        #model = model.half()
        model.to(device)
    else:
        num_gpu = torch.cuda.device_count()
        num_gpu = args.num_gpu
        model = load_model_on_gpus(model_dir, num_gpus=num_gpu, model_class=model_class)
    global_dict['model'] = model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Pad left for inferencing
    tokenizer.padding_side = 'left'
    global_dict['tokenizer'] = tokenizer
    print('global dict after', len(global_dict))

    if args.start_flask:
        if local_rank == 0:
            #start server
            print('starting server------------')
            model_name = os.path.basename(model_dir.rstrip(os.sep)) if args.serving_name is None else args.serving_name
            with open('/etc/hosts') as f:
                ip = [l.split()[0] for l in f if l.strip() != ''][-1]
                port = random.randint(8090, 10000)
                #port = 8123
                url = 'http://{}:{}/generate/'.format(ip, port)
            print(url)
            route_fn = os.path.expanduser('~/chatgpt_route.json')
            with open(route_fn) as f:
                model_meta = json.load(f)
            model_meta[model_name] = {'url': url}
            with open(route_fn, 'w') as f:
                f.write(json.dumps(model_meta, indent=5))

            if 'HFS' in os.environ:
                hfs = os.environ['HFS']
                hdfs_route_fn = os.environ['CCX_HDFS_HOME'] + '/spark/chatgpt/chatgpt_route.json'
                os.system(f'{hfs} -rm {hdfs_route_fn}')
                os.system(f'{hfs} -put {route_fn} {hdfs_route_fn}')
            # Single process for gpu thread safaty
            app.run(host='0.0.0.0', port=port, threaded=False, processes=1, debug=False)
            print('starting server done------------')
        else:
            infer_loop()

    else:
        # One observation: if the prompt ends with a space " ", there is a high chance that
        # the original model (without finetuning) will stuck and produce no response.
        # Finetuned models have less such issue. Thus following prompts all end with ":"
        # to make it a more meaningful comparison.
        if args.language == "English":
            prompts = [
                "Human: Explain the moon landing to a 6 year old in a few sentences. Assistant:",
                "Human: Write a short poem about a wise frog. Assistant:",
                "Human: Who was president of the United States in 1955? Assistant:",
                "Human: How does a telescope work? Assistant:",
                "Human: Why do birds migrate south for the winter? Assistant:"
            ]
        elif args.language == "Chinese":
            prompts = [
                "Human: Please tell me about Microsoft in a few sentence? Assistant:",
                "Human: 请用几句话介绍一下微软? Assistant:",
                "Human: 用几句话向6岁的孩子解释登月。 Assistant:",
                "Human: 写一首关于一只聪明的青蛙的短诗。 Assistant:",
                "Human: 谁是1955年的美国总统? Assistant:", "Human: 望远镜是如何工作的? Assistant:",
                "Human: 鸟类为什么要南迁过冬? Assistant:",
                "Human: 请描述一种可以用于识别恶意网站的机器学习算法 Assistant:",
                "Human: 计算5+6的和 Assistant:",
                "计算5+6的和",
                "Human: 帮我制定一个厦门旅游的攻略 Assistant:",
                "Human: 枚举 \"明天\" 的同义词 Assistant:",
                "Human: Please tell me about Microsoft in a few sentence? Assistant:",
                "Human: Explain the moon landing to a 6 year old in a few sentences. Assistant:",
                "Human: Write a short poem about a wise frog. Assistant:",
            ]

        for prompt in prompts:
            if 'chatglm' in args.model_name_or_path_baseline:
                print('chatglm')
                prompt_chatglm(args, model, tokenizer, device, prompt)
            else:
                prompt_eval(args, model, tokenizer, device, prompt)

if __name__ == "__main__":
    main()
