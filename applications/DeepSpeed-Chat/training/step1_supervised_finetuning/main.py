#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from megatron import mpu
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='6,2,2',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `6,2,2`'
                        'will use 60% of data for phase 1, 20% for phase 2'
                        'and 20% for phase 3.')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--tensor_parallel',
                        action='store_true',
                        help='tp')
    parser.add_argument('--pipeline_parallel',
                        action='store_true',
                        help='pp')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage, max_out_tokens=args.max_seq_len)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    print_rank_0('random seed set!')

    #assert not args.offload, "zero-offload is not currently supported but coming soon!"

    torch.distributed.barrier()
    print_rank_0('after barrier!')
    model_class = AutoModelForCausalLM
    tokenizer_class = AutoTokenizer 
    tok_params = {}
    if 'llama' in args.model_name_or_path:
        tok_params['fast_tokenizer'] = False
    if 'chatglm' in args.model_name_or_path:
        from chatglm_6b.tokenization_chatglm import ChatGLMTokenizer
        from chatglm_6b.modeling_chatglm import ChatGLMForConditionalGeneration
        tokenizer_class = ChatGLMTokenizer
        model_class = ChatGLMForConditionalGeneration

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, trust_remote_code=True, **tok_params)
    print_rank_0('load tokenizer done!')
    if tokenizer.pad_token is None:
        print(tokenizer.eos_token, tokenizer.eos_token_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(tokenizer.pad_token, tokenizer.pad_token_id)

    if args.pipeline_parallel:
        from llama_pipeline_model import get_model
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = get_model(config)
    else:
        model = create_hf_model(model_class,
                                args.model_name_or_path,
                                tokenizer,
                                ds_config,
                                disable_dropout=args.disable_dropout)
    print_rank_0("***** Model done*****", args.global_rank)

    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)

    # Prepare the data
    train_phase = 1
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_seq_len,
        sft_only_data_path=args.sft_only_data_path)
    print('train_num', len(train_dataset), 'eval_num', len(eval_dataset))
    print_rank_0("***** Dataset done*****", args.global_rank)

    init_params = {}

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)

    if args.tensor_parallel:
        mpu.initialize_model_parallel(
            tensor_model_parallel_size_=2,
            pipeline_model_parallel_size_=1,
        )
        init_params['mpu'] = mpu
        params = {
            'num_replicas': mpu.get_data_parallel_world_size(),
            'rank': mpu.get_data_parallel_rank(),
        }
        print('dd_params', params)
        train_sampler = DistributedSampler(train_dataset, **params)
        eval_sampler = DistributedSampler(eval_dataset, **params)


    if args.pipeline_parallel:
        from llama_pipeline_model import DataCollatorForPromptDataset
        data_collator = DataCollatorForPromptDataset()

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    print_rank_0("***** Dataloader done*****", args.global_rank)
    #for idx, e in enumerate(train_dataloader):
    #    if idx < 5:
    #        pass
            #print(idx, e)
            #for k in e.keys():
            #    print(k, e[k].shape)


    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)
    print_rank_0("***** Optim group done*****", args.global_rank)

    print_rank_0(len(train_dataloader), args.global_rank)
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))
    print_rank_0("***** Optim done*****", args.global_rank)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
    print_rank_0("***** Lr scheduler group done*****", args.global_rank)

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
        **init_params)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    #perplexity = evaluation(model, eval_dataloader)
    #print_rank_0(f"ppl: {perplexity}", args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        epoch_iterator = tqdm(train_dataloader, disable=args.global_rank != 0)
        save_per_step = 1000
        #loader = iter(deepspeed.utils.RepeatingLoader(train_dataloader))
        #for step in range(1000):
        for step, batch in enumerate(epoch_iterator):
            if step < 10:
                print_rank_0('to_device', args.global_rank)
            batch = to_device(batch, device)
            if step < 10:
                print_rank_0('after to_device', args.global_rank)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            """
            loss = model.train_batch(data_iter=loader)
            """

            epoch_iterator.set_postfix({'loss': loss.item()})  #
            model.backward(loss)
            model.step()

            if step > 0 and step % save_per_step == 0:
                if args.global_rank == 0:
                    save_hf_format(model, tokenizer, args)

                if args.zero_stage == 3:
                    # For zero stage 3, each gpu only has a part of the model, so we need a special save function
                    save_zero_three_model(model,
                                          args.global_rank,
                                          args.output_dir,
                                          zero_stage=args.zero_stage)

        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)

        # Evaluate perplexity on the validation set.
        print_rank_0(
            f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        #perplexity = evaluation(model, eval_dataloader)
        perplexity = 0.0
        print_rank_0(f"ppl: {perplexity}", args.global_rank)
        model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        print_rank_0('saved the final model to {}...'.format(args.output_dir), args.global_rank)
        #model = convert_lora_to_linear_layer(model)

        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)

        torch.distributed.barrier()
        print_rank_0('saved the final model to {}...'.format(args.output_dir), args.global_rank)

        if args.global_rank == 0:
            output_dir = args.output_dir + '_ds'
            model.save_checkpoint(output_dir)

            model = AutoModelForCausalLM.from_pretrained(args.output_dir)
            args.output_dir = args.output_dir + '_split'
            save_hf_format(model, tokenizer, args)
            model.save_pretrained(args.output_dir, max_shard_size='2GB')


        torch.distributed.barrier()


if __name__ == "__main__":
    main()

