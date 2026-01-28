set -x
ENGINE=${1:-vllm}

# 3. 设置严格的 Ray 隔离环境变量
nvidia-smi
nvcc -V
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export RAY_DEBUG=legacy
export HYDRA_FULL_ERROR=1 

DATA_PATH="xxxx.parquet"
OUTPUT_DIR="xxx"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +use_self_verify=true \
    +use_think_mode=true \
    +think_alpha=0.2 \
    +lambda_consistency=0.2 \
    +FORMAT_PENALTY=0.1 \
    +LENGTH_THRESHOLD=1024 \
    +LEN_PENALTY_COEFF=0.002 \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    data.train_batch_size=16 \
    data.max_prompt_length=5120 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.prompt_key=prompt \
    data.image_key=image \
    actor_rollout_ref.model.path=xxxx \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.03 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=6144 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.max_pixels=451584 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='xxx \
    trainer.experiment_name='xxx' \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 $@
