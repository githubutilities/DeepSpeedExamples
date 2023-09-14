import os
from typing import Dict, Tuple, Union, Optional

import torch
from torch.nn import Module
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM

def auto_configure_llama_device_map_disk(num_gpus=8) -> Dict[str, int]:
    device_map = {
        'model.embed_tokens': 'disk',
        'model.norm.weight': 'disk',
        'lm_head.weight': 'disk',
    }
    num_trans_layers = 80
    per_gpu_layers = (num_trans_layers + 2) / num_gpus

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'model.layers.{i}'] = 'disk'
        used += 1

    return device_map



def auto_configure_llama_device_map(num_gpus: int) -> Dict[str, int]:
    device_map = {
        'model.embed_tokens': 0,
        'model.norm.weight': num_gpus - 1,
        'lm_head.weight': 0,
    }
    num_trans_layers = 80
    per_gpu_layers = (num_trans_layers + 2) / num_gpus

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'model.layers.{i}'] = gpu_target
        used += 1

    return device_map


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {'transformer.word_embeddings': 0,
                  'transformer.final_layernorm': 0, 'lm_head': 0}

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.layers.{i}'] = gpu_target
        used += 1

    device_map.update(auto_configure_llama_device_map(num_gpus))
    return device_map


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2,
                       device_map: Optional[Dict[str, int]] = None, model_class=None, **kwargs) -> Module:
    if model_class is None:
        model_class = AutoModelForCausalLM
    kwargs['torch_dtype'] = torch.float16
    kwargs['offload_state_dict'] = True
    kwargs['trust_remote_code'] = True
    try:
        config = AutoConfig.from_pretrained(checkpoint_path)
        if True:
            model = model_class.from_pretrained(checkpoint_path, **kwargs).half()
        else:
            config = AutoConfig.from_pretrained(checkpoint_path, **kwargs)
            model = model_class.from_config(config).half()
    except:
        from alpaca_farm.models.reward_model import RewardConfig
        from alpaca_farm.models.reward_model import RewardModel
        config = RewardConfig.from_pretrained(checkpoint_path)
        critic_model = RewardModel.from_pretrained(checkpoint_path, 
            flash_attn=False,
            fp16=True,
            bf16=False,
            low_cpu_mem_usage=True,
            device_map=None,
            config=config,
        )
        model = critic_model
    model.eval()

    if num_gpus < 2 and device_map is None:
        #model = model_class.from_pretrained(checkpoint_path, **kwargs).half().cuda()
        model = model.cuda()
    elif True:
        import deepspeed
        import deepspeed.module_inject as module_inject
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        if False:
            config = AutoConfig.from_pretrained(checkpoint_path, **kwargs)
            with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
                model = model_class.from_config(config)
        else:
            if False:
                kwargs['device_map'] = auto_configure_llama_device_map_disk()
                kwargs['offload_folder'] = '/tmp/offload'
        injection_policy = { LlamaForCausalLM: module_inject.replace_policy.LLAMALayerPolicy }
        world_size = num_gpus
        print(world_size, 'world_size')
        #zero = deepspeed.runtime.zero.config.DeepSpeedZeroConfig(stage=3)
        config = {
            'dtype': 'fp16',
            'zero': {
                'stage': 3,
                'offload_param': {
                    'device': 'cpu',
                },
            },
            #'injection_policy': injection_policy,
            'injection_policy_tuple': (injection_policy, ),
        }
        if False:
            init_fn = deepspeed.initialize
            init_kwargs = {}
        else:
            init_fn = deepspeed.init_inference
            init_kwargs = {
                'config': config,
                'mp_size': world_size,
            }
        model = init_fn(
            model=model,
            #config=config,
            #mp_size=world_size,
            #dtype=torch.float16,
            #injection_policy=injection_policy,
            #replace_with_kernel_inject=True,
            **init_kwargs,
        )
    else:
        from accelerate import init_empty_weights
        from accelerate import dispatch_model, load_checkpoint_and_dispatch

        if True:
            if device_map is None:
                device_map = auto_configure_device_map(num_gpus)
            kwargs['device_map'] = device_map
            #kwargs['device_map'] = 'sequential'
            #kwargs['max_memory'] = '32GB'
            kwargs['torch_dtype'] = torch.float16
            model = model_class.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs)

            model = dispatch_model(model, device_map=device_map)
        else:
            config = AutoConfig.from_pretrained(checkpoint_path)

            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)

            device_map = auto_configure_device_map(num_gpus)

            model = load_checkpoint_and_dispatch(
                model, checkpoint_path, device_map=device_map,
            )

    return model


