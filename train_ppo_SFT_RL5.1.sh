
#适用4A800
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR='/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/dataset/legal_exam'
WAND_PROJECT='Search-R1'
# === 原始 SFT 模型路径 (用于初始化架构和Tokenizer) ===
# 保持原有的 SFT 路径不变
export BASE_MODEL="/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/SFT_ckp/qwen3_SFT5.5_0127/checkpoint-2-504/tfmr"

# === [核心修改] 未合并的 Checkpoint 路径 ===
# 指向您要恢复的具体 Step 文件夹
# 结构通常是: .../experiment_name/global_step_xxx
export RESUME_CHECKPOINT="/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/RL_ckp/legal_exam-ppo-qwen3-8b-RL-5.0-0203/global_step_120"

export EXPERIMENT_NAME=legal_exam-ppo-qwen3-8b-RL-5.1-0205

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    +data.train_data_num=null \
    +data.val_data_num=null \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=25900 \
    data.max_response_length=1500 \
    +data.max_start_length=1000 \
    +data.max_obs_length=700 \
    +data.shuffle_train_dataloader=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=30000 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.temperature=0.2 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=true \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_mini_batch_size=16 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.strategy=fsdp \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    +algorithm.no_think_rl=false \
    reward_model.enable=false \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.val_only=false \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=10 \
    trainer.total_training_steps=300 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=$RESUME_CHECKPOINT \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/RL_ckp/$EXPERIMENT_NAME \
    +max_turns=10 \
    ray_kwargs.ray_init.num_cpus=16 \
    +retriever.url="http://127.0.0.1:8007/retrieve" \
    +retriever.topk=10 \
    2>&1 | tee $EXPERIMENT_NAME.log