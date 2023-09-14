#!/bin/bash
cd "$(dirname "${BASH_SOURCE}")"/..;
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model

ps -ef | grep trainer | grep -v grep | tee /dev/stderr | awk '{print $2}' | xargs kill -9
set -x
source ./utils/env.sh

bash ./scripts/web_demo.sh &
#python -m accelerate.commands.accelerate_cli launch \
#python \
MODEL_NAME_OR_PATH=$1
cuda_start_id=${2:-"0"}
num_gpu=${3:-"4"}
visible_device=`seq -s',' $cuda_start_id $(($cuda_start_id + $num_gpu - 1))`
echo $visible_device
args="--start_flask"
port=$(echo $((20000 + $RANDOM % 1000)))
#    --num_gpus $num_gpu \
launch_cmd="python -m deepspeed.launcher.runner "
$launch_cmd \
    --include localhost:$visible_device \
    --master_addr 127.0.0.1 \
    --master_port $port \
    ./utils/inference_server.py \
    --num_gpu $num_gpu \
    --model_name_or_path_baseline $MODEL_NAME_OR_PATH $args ${@:4}

