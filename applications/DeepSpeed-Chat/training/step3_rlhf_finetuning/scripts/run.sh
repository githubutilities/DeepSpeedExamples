#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
cd "$(dirname "${BASH_SOURCE}")"/..;
source ./utils/env.sh
set -x
setup_train_env

ACTOR_MODEL_PATH=${STEP3_ACTOR_DIR:-"llama-7b_sft_zh_gpt4_2w_ws16_bs4_seq2048_sft_1plus1_fifth_ws8_bs2_acc8_seq2048"}
CRITIC_MODEL_PATH=${STEP3_CRITIC_DIR:-"llama-7b_sft_zh_1plus1_fifth_reward"}
ACTOR_ZERO_STAGE=3
CRITIC_ZERO_STAGE=3

Actor_Lr=9.65e-6
Critic_Lr=5e-6
#Actor_Lr=5e-6
#Critic_Lr=2e-6

STEP_BATCH_SIZE=${STEP3_STEP_BATCH_SIZE:-256}
STEP_PER_DEVICE_BATCH_SIZE=${STEP3_STEP_PER_DEVICE_BATCH_SIZE:-1}
ROLLOUT_BATCH_SIZE=${STEP3_ROLLOUT_BATCH_SIZE:-128}

per_device_train_batch_size=$(($ROLLOUT_BATCH_SIZE / $WORLD_SIZE))
gradient_accumulation_steps=$(($STEP_BATCH_SIZE / $STEP_PER_DEVICE_BATCH_SIZE / $WORLD_SIZE))
DATA_DIR=${STEP3_DATASET_NAME:-"`readlink -f ~/data/chatgpt/sft_1plus1_fifth_reward/`"}
OUTPUT=${STEP3_OUTPUT_DIR:-"./output_ppo"}
mkdir -p $OUTPUT
   #--data_path Dahoas/rm-static \
$launch_prefix_cmd \
    main.py \
   --data_path $DATA_DIR \
   --data_split 0,0,10 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size $per_device_train_batch_size \
   --per_device_mini_train_batch_size $STEP_PER_DEVICE_BATCH_SIZE \
   --generation_batch_numbers 1 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 3 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps $gradient_accumulation_steps \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --tp_gather_partition_size 4 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --release_inference_cache \
   --disable_actor_dropout \
   --output_dir $OUTPUT \
   --inference_tp_size 4 \
   --enable_hybrid_engine $STEP3_EXTRA_ARGS \
   2>&1 | \
   tee $OUTPUT/training_`get_distributed_rank`.log

   #--offload_reference_model \
   #--actor_lora_dim 128 \
   #--actor_lora_module_name decoder.layers. \

