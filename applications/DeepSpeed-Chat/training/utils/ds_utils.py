# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4


def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        initial_scale_power=3):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            #"device": device,
            "device": 'none',
            "pin_memory": False,
        },
        "offload_optimizer": {
            "device": device,
            "pin_memory": False,
        },
        "stage3_param_persistence_threshold": 1e4,

        #"overlap_comm": True,
        #"allgather_bucket_size": 5e8,
        #"reduce_scatter": True,
        #"contiguous_gradients": True,
        #"reduce_bucket_size": 5e8,
        #"sub_group_size": 3e7,
        #"reduce_bucket_size": "auto",
        #"stage3_param_persistence_threshold": "auto",
        #"stage3_max_reuse_distance": 3e7,

        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": True,
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        #"train_micro_batch_size_per_gpu": "auto",
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "amp": {
            "enabled": False,
        },
        #"bf16": {
        #    "enabled": True,
        #},
        "fp16": {
            "enabled": True,
            "initial_scale_power": initial_scale_power,
            "loss_scale_window": 100
        },
        "autotuning": {
            "enabled": False,
            "mp_size": 2,
            "arg_mappings": {
                #"train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
            },
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        }
    }


def get_eval_ds_config(offload, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
