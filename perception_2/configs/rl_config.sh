#!/bin/bash

########################
# * Model
########################
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
RM_MODEL_PATH=Qwen/Qwen3-VL-32B-Instruct

########################
# Data
########################
TRAIN_FILE=_preprocessed_dataset/dataset_mixed_22k_qwen3-vl-8b_20260218_205144_filtered_PROCESSED_NEW_SYSTEM_PROMPT/train.parquet
VAL_FILE=_preprocessed_dataset/dataset_mixed_22k_qwen3-vl-8b_20260218_205144_filtered_PROCESSED_NEW_SYSTEM_PROMPT/test.parquet

TRAIN_FILE_30K=_preprocessed_dataset/dataset_mixed_22k_qwen3-vl-8b_20260302_162334_filtered_NEW_SYSTEM_PROMPT_PROCESSED/train.parquet
VAL_FILE_30K=_preprocessed_dataset/dataset_mixed_22k_qwen3-vl-8b_20260218_205144_filtered_PROCESSED_NEW_SYSTEM_PROMPT/test.parquet

TRAIN_BATCH_SIZE=128 # 可能调大
MAX_PROMPT_LENGTH=5500
MAX_RESPONSE_LENGTH=1500

########################
# PPO / GRPO
########################
LR=1e-6
ROLLOUT_N=8

PPO_MINI_BATCH_SIZE=64 #可能调大
PPO_MICRO_BATCH_SIZE=16 #可能调大

KL_COEF=0.008 #调大或者调小

########################
# vLLM rollout
########################
GPU_MEMORY_UTIL=0.65
RM_GPU_MEMORY_UTIL=0.7
TENSOR_PARALLEL_SIZE=2 # 可能调小

########################
# Training
########################
EPOCHS=5
SAVE_FREQ=70
TEST_FREQ=3

########################
# Trigger schedule
########################

# Ablation Study, 后续再做
TRIGGER_WARMUP_RATIO=0.3
TRIGGER_START_RATIO=0.6
TRIGGER_FINAL_RATIO=0.5

########################
# Logging
########################
PROJECT_NAME=verl_correction
EXP_NAME=qwen2_5_vl_7b_mix_10K_no_correction_n8

########################
# Paths
########################
TRIGGER_FILE=./perception/utils/triggers.json

ROLLOUT_SAVE_PATH=./rollout_saved/temp

