



#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
source /home/qspace/data/ruiwen/.bashrc
source /home/qspace/data/ruiwen/net.sh
# DeepSpeed Team
#OUTPUT=$1
#ZERO_STAGE=$2
#if [ "$OUTPUT" == "" ]; then
#    OUTPUT=./output
#fi
#if [ "$ZERO_STAGE" == "" ]; then
#    ZERO_STAGE=2
#fi
#mkdir -p $OUTPUT
#chmod -R 777 $OUTPUT


ZERO_STAGE=2
BATCH_SIZE=4
OUTPUT=/mnt/yardcephfs/mmyard/g_wxg_fd_search/ruiwen/projects/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output
MODEL_NAME_OR_PATH=/mnt/yardcephfs/mmyard/g_wxg_fd_search/ruiwen/projects/chat2/pretrained_models/llama-7b   #/mnt/yardcephfs/mmyard/g_wxg_fd_search/ruiwen/projects/chat2

#/Users/ruiwen/Documents/gpt/gpt2
#
deepspeed main.py \
   --data_path Dahoas/rm-static \
   --data_split 8,1,1 \
   --model_name_or_path $MODEL_NAME_OR_PATH \
   --per_device_train_batch_size $BATCH_SIZE \
   --per_device_eval_batch_size $BATCH_SIZE \
   --max_seq_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0.1 \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 42 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
#   > $OUTPUT/training.log


#   --lora_dim 128 \
#   --lora_module_name decoder.layers. \