import os
import sys
import argparse

from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForCausalLM
from model.reward_model import RewardModel

def _main():
    parser = argparse.ArgumentParser(description='model_split')
    parser.add_argument('-i', '--model_name', type=str, required=True)
    parser.add_argument('-o', '--output_name', type=str, default=None)
    parser.add_argument('--num_layer', type=int, required=True)
    args = parser.parse_args()
    model_name = args.model_name
    output_name = model_name.rstrip(os.sep) + '_split' if args.output_name is None else args.output_name
    config = AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = args.num_layer
    m = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    m.save_pretrained(output_name, max_shard_size='5GB')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_name)

if __name__ == '__main__':
    _main()

