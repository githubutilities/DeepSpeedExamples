# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
)

from transformers.deepspeed import HfDeepSpeedConfig

from .reward_model import RewardModel


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    disable_dropout=False):
    config_class = AutoConfig
    if 'chatglm' in model_name_or_path:
        from chatglm_6b.configuration_chatglm import ChatGLMConfig
        config_class = ChatGLMConfig
    else:
        config_class = AutoConfig
    model_config = config_class.from_pretrained(model_name_or_path, trust_remote_code=True)
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            trust_remote_code=True,
            config=model_config,
            #offload_state_dict=True,
            low_cpu_mem_usage=True,
            device_map={"": torch.cuda.current_device()},
        )
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            trust_remote_code=True,
            config=model_config,
            offload_state_dict=True,
        )

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8, for better performance

    return model


def create_critic_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        disable_dropout=False,
                        raw_init=False,
                        alpaca_reward_model=False):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule
    from deepspeed.runtime.utils import see_memory_usage
    if alpaca_reward_model:
        from alpaca_farm.models.reward_model import RewardConfig
        from alpaca_farm.models.reward_model import RewardModel as RewardModelAlpaca
        config = RewardConfig.from_pretrained(os.path.join(model_name_or_path, 'config.json.alpaca'))
        critic_model = RewardModel.from_pretrained(
            model_name_or_path, 
            config=config, 
            tokenizer=tokenizer,
            #offload_state_dict=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            #device_map={"": torch.cuda.current_device()},
        )
        #critic_model = RewardModelAlpaca.from_pretrained(model_name_or_path,
        #    flash_attn=False,
        #    fp16=True,
        #    bf16=False,
        #    low_cpu_mem_usage=True,
        #    device_map=None,
        #    config=config,
        #)
    elif raw_init:
        from .reward_model_bak import RewardModelBak
        critic_model = create_hf_model(AutoModelForCausalLM, model_name_or_path, tokenizer,
                                       ds_config, rlhf_training, disable_dropout)
        config = AutoConfig.from_pretrained(model_name_or_path)
        critic_model = RewardModelBak(
            #config=config,
            base_model=critic_model,
            tokenizer=tokenizer,
            num_padding_at_beginning=num_padding_at_beginning)
    else:
        critic_model = RewardModel.from_pretrained(
            model_name_or_path,
            tokenizer=tokenizer,
            num_padding_at_beginning=num_padding_at_beginning,
        )

    #if rlhf_training:
    if False:
        # critic model needs to load the weight here
        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"
        critic_model.load_state_dict(
            torch.load(model_ckpt_path, map_location='cpu'),
            strict=False,
        )
        print('after load')

    return critic_model
