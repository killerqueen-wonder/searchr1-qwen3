#适用4H200
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR='/data/panghuaiwen/legal_R1/dataset/RL_parquet'

export BASE_MODEL="/data/panghuaiwen/legal_R1/model/SFT_ckp/qwen3_SFT6.0_0407/checkpoint-2-708/tfmr"
export EXPERIMENT_NAME=legal_exam-ppo-qwen3-8b-RL-7.0-0415

WAND_PROJECT='Search-R1'

export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600
set -x
# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=28000 \
    data.max_response_length=1200 \
    +data.max_start_length=4000 \
    +data.max_obs_length=2000 \
    +data.shuffle_train_dataloader=true \
    data.filter_overlong_prompts=true \
    data.truncation='error' \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.temperature=0.2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=33000 \
    algorithm.use_kl_in_reward=false \
    +algorithm.no_think_rl=false \
    +trainer.use_critic=false \
    +trainer.do_search=true \
    trainer.critic_warmup=0 \
    trainer.logger='["wandb"]' \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=151 \
    trainer.resume_mode=disable \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/data/panghuaiwen/legal_R1/model/RL_ckp/$EXPERIMENT_NAME \
    +max_turns=10 \
    ray_kwargs.ray_init.num_cpus=16 \
    +retriever.url="http://127.0.0.1:8005/retrieve" \
    +retriever.topk=8 \
    2>&1 | tee $EXPERIMENT_NAME.log