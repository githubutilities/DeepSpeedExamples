# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import time
import torch
import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import AutoModelForCausalLM, get_scheduler

from utils.ds_utils import get_train_ds_config, get_eval_ds_config
from utils.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model, create_critic_model
from utils.utils import get_optimizer_grouped_parameters
"""
TODOs:
  * support HF models for critic (for debugging), must be a previously saved ckpt from step-2
  * determine ds_config/zero_stage based on model size, gpu style, world size, etc
    - get model size by creating simple meta model
    - 1.3b: zero-2 for actor/ref models, zero-0 for others
    - 13b+: zero-3 for all models
"""


def log_init(model_name, stime=None):
    if torch.distributed.get_rank() == 0:
        tag = "start" if stime is None else "end"
        suffix = "ing" if stime is None else "ed"
        duration = ""
        if stime is not None:
            duration = "(duration: {:.2f}s)".format(time.time() - stime)
        msg = f"[{tag}] Initializ{suffix} {model_name} Model [{tag}] {duration}"
        stars = (90 - len(msg)) // 2
        extra_star = "*" if (90 - len(msg)) % 2 == 1 else ""
        print("*" * stars + msg + "*" * stars + extra_star)
        return time.time()


class DeepSpeedRLHFEngine():

    def __init__(self, actor_model_name_or_path, critic_model_name_or_path,
                 tokenizer, args, num_total_iters):
        self.args = args
        self.num_total_iters = num_total_iters
        self.tokenizer = tokenizer

        self.actor = self._init_actor(
            actor_model_name_or_path=actor_model_name_or_path)
        self.ref = self._init_ref(
            actor_model_name_or_path=actor_model_name_or_path)
        self.actor_ema = None
        if self.args.enable_ema:
            self.actor_ema = self._init_ema(
                actor_model_name_or_path=actor_model_name_or_path)

        self.critic = self._init_critic(
            critic_model_name_or_path=critic_model_name_or_path)
        self.reward = self._init_reward(
            critic_model_name_or_path=critic_model_name_or_path)
        if self.args.critic_gradient_checkpointing:
            self.critic.gradient_checkpointing_enable()

    def _init_actor(self, actor_model_name_or_path):
        stime = log_init("Actor")

        # DS Config
        ds_config = get_train_ds_config(
            offload=self.args.offload,
            stage=self.args.actor_zero_stage,
            enable_hybrid_engine=self.args.enable_hybrid_engine,
            inference_tp_size=self.args.inference_tp_size,
            release_inference_cache=self.args.release_inference_cache,
            pin_parameters=(not self.args.unpin_actor_parameters),
            tp_gather_partition_size=self.args.tp_gather_partition_size,
            max_out_tokens=self.args.max_prompt_seq_len +
            self.args.max_answer_seq_len,
            #initial_scale_power=0,
            )
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_mini_train_batch_size
        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor

        # Model
        actor_model = create_hf_model(
            model_class=AutoModelForCausalLM,
            model_name_or_path=actor_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_config,
            disable_dropout=self.args.disable_actor_dropout)

        # LoRA
        if self.args.actor_lora_dim > 0:
            actor_model = convert_linear_layer_to_lora(
                actor_model, self.args.actor_lora_module_name,
                self.args.actor_lora_dim)
            if self.args.only_optimize_lora:
                actor_model = only_optimize_lora_parameters(actor_model)

        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(
            actor_model, self.args.actor_weight_decay)
        optim = AdamOptimizer(optim_params,
                              eps=1e-5,
                              lr=self.args.actor_learning_rate,
                              betas=(0.9, 0.95))

        # LR Scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        #TODO: move enable_hybrid_engine and pin_parameters to ds_config
        actor_engine, *_ = deepspeed.initialize(model=actor_model,
                                                optimizer=optim,
                                                lr_scheduler=lr_scheduler,
                                                config=ds_config)

        log_init("Actor", stime=stime)

        return actor_engine

    def _init_ref(self, actor_model_name_or_path):
        stime = log_init("Ref")
        # DS Config
        zero_stage = self.args.actor_zero_stage
        if zero_stage != 3:
            # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory for ref model
            zero_stage = 0
        ds_config = get_eval_ds_config(self.args.offload_reference_model,
                                       zero_stage)
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_mini_train_batch_size
        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor

        ref_model = create_hf_model(AutoModelForCausalLM,
                                    actor_model_name_or_path, self.tokenizer,
                                    ds_config)
        if True:
            if False:
                import deepspeed.module_inject as module_inject
                from transformers.models.llama.modeling_llama import LlamaForCausalLM
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                injection_policy = { LlamaForCausalLM: (module_inject.replace_policy.LLAMALayerPolicy, ) }
                #injection_policy = { LlamaDecoderLayer: ('self_attn.o_proj', 'mlp.down_proj')}
                injection_policy = { LlamaDecoderLayer: ('mlp.down_proj')}
                ds_config = {
                    'dtype': 'fp16',
                    'zero': {
                        'stage': 3,
                        #'offload_param': {
                        #    'device': 'cpu',
                        #},
                    },
                    "tensor_parallel": {"tp_size": 4},
                    'replace_with_kernel_inject': True,
                    #'injection_policy': injection_policy,
                    #'injection_policy_tuple': (injection_policy, ),
                }

                ref_engine = deepspeed.init_inference(
                    model=ref_model,
                    config=ds_config,
                )
            else:
                ref_engine, *_ = deepspeed.initialize(
                    model=ref_model,
                    config=ds_config,
                )

        else:
            class ref_engine:
                module = None
                def eval():
                    pass
            ref_engine.module = ref_model

        log_init("Ref", stime=stime)
        return ref_engine

    def _init_ema(self, actor_model_name_or_path):
        stime = log_init("EMA")
        # DS Config
        zero_stage = self.args.actor_zero_stage
        if zero_stage != 3:
            # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            zero_stage = 0
        ds_config = get_eval_ds_config(self.args.offload_reference_model,
                                       zero_stage)
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_mini_train_batch_size
        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor

        actor_model_ema = create_hf_model(AutoModelForCausalLM,
                                          actor_model_name_or_path,
                                          self.tokenizer, ds_config)
        if self.args.actor_lora_dim > 0:
            actor_model_ema = convert_linear_layer_to_lora(
                actor_model_ema, self.args.actor_lora_module_name,
                self.args.actor_lora_dim)

        ema_engine, *_ = deepspeed.initialize(model=actor_model_ema,
                                              config=ds_config)

        log_init("EMA", stime=stime)
        return ema_engine

    def _init_critic(self, critic_model_name_or_path):
        stime = log_init("Critic")
        ds_config = get_train_ds_config(offload=self.args.offload,
                                        stage=self.args.critic_zero_stage)
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_mini_train_batch_size
        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps

        #TODO(jeff): should not be needed, we should be able to use ds_config above
        #TODO(jeff): it means we never create the critic w. zero.init context if we are using ZeRO-3
        ds_eval_config = get_eval_ds_config(offload=False, stage=0)

        # Model
        critic_model = create_critic_model(
            model_name_or_path=critic_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_eval_config,
            num_padding_at_beginning=self.args.num_padding_at_beginning,
            rlhf_training=True,
            disable_dropout=self.args.disable_critic_dropout,
            alpaca_reward_model=self.args.load_alpaca_reward_model,
        )

        # LoRA
        if self.args.critic_lora_dim > 0:
            critic_model = convert_linear_layer_to_lora(
                critic_model, self.args.critic_lora_module_name,
                self.args.critic_lora_dim)
            if self.args.only_optimize_lora:
                critic_model = only_optimize_lora_parameters(critic_model)

        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        optim_pararms = get_optimizer_grouped_parameters(
            critic_model, self.args.critic_weight_decay)
        optim = AdamOptimizer(optim_pararms,
                              lr=self.args.critic_learning_rate,
                              betas=(0.9, 0.95))

        # LR Scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        critic_engine, *_ = deepspeed.initialize(model=critic_model,
                                                 optimizer=optim,
                                                 lr_scheduler=lr_scheduler,
                                                 config=ds_config)

        log_init("Critic", stime=stime)
        return critic_engine

    def _init_reward(self, critic_model_name_or_path):
        stime = log_init("Reward")
        # DS Config
        zero_stage = self.args.critic_zero_stage
        if zero_stage != 3:
            # If critic is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            zero_stage = 0

        ds_config = get_eval_ds_config(offload=self.args.offload,
                                       stage=zero_stage)
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_mini_train_batch_size
        ds_config[
            'train_batch_size'] = self.args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps

        #TODO(jeff): should not be needed, we should be able to use ds_config above
        #TODO(jeff): it means we never create the critic w. zero.init context if we are using ZeRO-3
        ds_eval_config = get_eval_ds_config(offload=False, stage=0)

        # Model
        reward_model = create_critic_model(
            model_name_or_path=critic_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_eval_config,
            num_padding_at_beginning=self.args.num_padding_at_beginning,
            rlhf_training=True,
            alpaca_reward_model=self.args.load_alpaca_reward_model,
        )

        reward_engine, *_ = deepspeed.initialize(model=reward_model,
                                                 config=ds_config)

        log_init("Reward", stime=stime)
        return reward_engine
