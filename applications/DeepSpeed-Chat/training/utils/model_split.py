import os
import sys
import argparse

from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForCausalLM
from model.reward_model import RewardModel

def _main():
    parser = argparse.ArgumentParser(description='model_split')
    parser.add_argument('-i', '--model_name', type=str, required=True)
    parser.add_argument('-o', '--output_name', type=str, default=None)
    parser.add_argument('--split_size', type=str, default='2GB')
    parser.add_argument('--step1', action='store_true', default=False)
    parser.add_argument('--step2', action='store_true', default=False)
    parser.add_argument('--alpaca', action='store_true', default=False)
    args = parser.parse_args()
    model_name = args.model_name
    output_name = model_name.rstrip(os.sep) + '_split' if args.output_name is None else args.output_name
    kwargs = {
        'trust_remote_code': True,
    }
    if args.alpaca:
        from alpaca_farm.models.reward_model import RewardConfig
        config = RewardConfig.from_pretrained(os.path.join(model_name, 'config.json.alpaca'), **kwargs)
        print(config)
        if not os.path.exists(output_name):
            os.makedirs(output_name)
        config.to_json_file(os.path.join(output_name, 'config.json.alpaca'))
    else:
        config = AutoConfig.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    #m = RewardModel(config=config, tokenizer=tokenizer, base_model=model)
    print('spliting model...')
    if args.step1:
        m = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    elif args.step2:
        m = RewardModel.from_pretrained(model_name, tokenizer=tokenizer, config=config, **kwargs)
    else:
        raise NotImplementedError
    m.save_pretrained(output_name, max_shard_size=args.split_size)
    tokenizer.save_pretrained(output_name)

if __name__ == '__main__':
    _main()

