set -x
ENGINE=${1:-vllm}

source "$(dirname "$0")/../configs/rl_config.sh"
export $(grep -v '^#' ../configs/reward_function_hyperparam.env | xargs)


DATE=$(date +%m%d)
EXP_NAME="${DATE}_base_run"
ROLLOUT_SAVE_PATH = "./rollouts_saved/${EXP_NAME}"

VLLM_USE_V1=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    reward_model.enable=True \
    reward_model.model.path=$RM_MODEL_PATH \
    reward_model.rollout.gpu_memory_utilization=$RM_GPU_MEMORY_UTIL \
    reward_model.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.total_epochs=$EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    +trainer.trigger_warmup_ratio=$TRIGGER_WARMUP_RATIO \
    +trainer.trigger_start_ratio=$TRIGGER_START_RATIO \
    +trainer.trigger_final_ratio=$TRIGGER_FINAL_RATIO \
    +trainer.trigger_file_path=$TRIGGER_FILE \
    trainer.validation_data_dir="${ROLLOUT_SAVE_PATH}/val" \
    trainer.rollout_data_dir="${ROLLOUT_SAVE_PATH}/train" \
    custom_reward_function.path=perception_2/reward_function_loop.py \
    custom_reward_function.name=compute_score \
    $@