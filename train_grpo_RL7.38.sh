#!/bin/bash
# ==============================================================================
# RAG + LLM Judge + GRPO 全自动训练总管道脚本 (RL 7.35)
# 适用环境：4 x A800 (80GB), Kubernetes / AI Studio
# 显存隔离方案 FSDP + TP=2 + FlashAttention + FP8(后台)

# ==============================================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR='/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/dataset/multi_form_RL/with_search'
export BASE_MODEL="/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/SFT_ckp/qwen3_SFT6.2_0508/checkpoint-2-750/tfmr"

export REWARD_MODEL="/F00120250029/lixiang_share/Models/Qwen3-8B"
export FILTER_MODEL="/F00120250029/lixiang_share/Models/Qwen3-8B"

export EXPERIMENT_NAME=legal_exam-ppo-qwen3-8b-RL-7.3-A800-fp8-lora
export WAND_PROJECT='Search-R1'
export WANDB_API_KEY='847a7dd2aadbd8146fa82d3cc3b88826530401ec'

# 端口配置
RETRIEVER_PORT=8005
VLLM_PORT=8006
REWARD_PORT=9000 

# 超时与分布式环境设置
PORT_TIMEOUT=1800 
export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600
export NCCL_SHM_DISABLE=1
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

# conda 初始化
export CONDA_SH="/F00120250029/lixiang_share/Data/conda/etc/profile.d/conda.sh"

info() { echo -e "\033[32m[INFO]\033[0m $1"; }
error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

wait_for_port() {
    local port=$1
    local timeout=$2
    local session_name=$3
    local start=$(date +%s)
    info "等待端口 $port 就绪（超时 ${timeout}s）..."
    
    while true; do
        if (echo > /dev/tcp/127.0.0.1/$port) >/dev/null 2>&1; then
            info "端口 $port 已就绪"
            return 0
        fi
        local now=$(date +%s)
        if [ $((now - start)) -ge $timeout ]; then
            error "端口 $port 未在 ${timeout}s 内就绪。查看日志: tmux attach -t $session_name"
            return 1
        fi
        sleep 5
    done
}

kill_session() {
    if tmux has-session -t "$1" 2>/dev/null; then
        info "清理历史会话: $1"
        tmux kill-session -t "$1"
        sleep 1
    fi
}

tmux_send_commands() {
    local session=$1
    shift
    tmux send-keys -t "$session" "source $CONDA_SH" C-m
    for cmd in "$@"; do
        tmux send-keys -t "$session" "$cmd" C-m
    done
}

if ! command -v tmux &> /dev/null; then
    info "未检测到 tmux，尝试安装..."
    apt update && apt install -y tmux
fi

# ==================== 1. 启动底层依赖服务 (后台 tmux) ====================
info "开始拉起底层服务 (A800 资源压榨版)..."

# # 组件 A: Reward LLM (GPU 2)
# kill_session "reward_llm"
# tmux new-session -d -s reward_llm -n reward
# tmux_send_commands "reward_llm" \
#     "export CUDA_VISIBLE_DEVICES=2" \
#     "conda activate vllm_server" \
#     "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH" \
#     "python -m vllm.entrypoints.openai.api_server \
#         --model $REWARD_MODEL \
#         --served-model-name qwen3-8b-reward \
#         --host 0.0.0.0 --port $REWARD_PORT \
#         --enable-prefix-caching \
#         --max-num-seqs 32 \
#         --max-model-len 32000 \
#         --gpu-memory-utilization 0.8 \
#         --kv-cache-dtype fp8_e5m2 \
#         --trust-remote-code"

# # 组件 B: RAG 检索过滤服务 (GPU 3)
# kill_session "retriever_filter8005"
# tmux new-session -d -s retriever_filter8005 -n retriever
# tmux_send_commands "retriever_filter8005" \
#     "conda activate retriever_filter" \
#     "export TRANSFORMERS_CACHE=/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model" \
#     "export HF_HUB_CACHE=/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model" \
#     "export TRANSFORMERS_OFFLINE=1" \
#     "export HF_HUB_OFFLINE=1" \
#     "cd /F00120250029/lixiang_share/panghuaiwen_share/legal_R1/searchr1-qwen3" \
#     "git pull origin main" \
#     "python search_r1/search/async_retrieval_server.py \
#         --port $RETRIEVER_PORT \
#         --corpus_path '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/dataset/law/法律法规3.0.jsonl' \
#         --case_corpus_path '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/dataset/case/lecard_court_psi.jsonl' \
#         --retriever_name hybrid_filter \
#         --dictionary_path '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/dataset/dictionary/THUOCL_law.txt' \
#         --search_depth 5 --bm25_weight 15 --bm25_weight_factor 2 --bm25_k1 0.15 --bm25_b 0.35 --topk 8 \
#         --retriever_model 'shibing624/text2vec-base-chinese-paraphrase' \
#         --filter_model $FILTER_MODEL \
#         --gpu_ids 3 --gpu_memory_limit_per_gpu 10"

# # 组件 C: Filter/Rerank 服务 (GPU 3 的剩余空间或 CPU)
# kill_session "vllm"
# tmux new-session -d -s vllm -n vllm
# tmux_send_commands "vllm" \
#     "export CUDA_VISIBLE_DEVICES=3" \
#     "conda activate vllm_server" \
#     "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH" \
#     "python -m vllm.entrypoints.openai.api_server \
#         --model $FILTER_MODEL \
#         --served-model-name Qwen3-8B \
#         --port $VLLM_PORT \
#         --max-model-len 12000 \
#         --gpu-memory-utilization 0.7 \
#         --kv-cache-dtype fp8_e5m2 \
#         --trust-remote-code"

# # ==================== 2. 健康检查 ====================
# info "等待所有依赖服务就绪..."

# wait_for_port $REWARD_PORT $PORT_TIMEOUT "reward_llm" || exit 1
# wait_for_port $VLLM_PORT $PORT_TIMEOUT "vllm" || exit 1
# wait_for_port $RETRIEVER_PORT $PORT_TIMEOUT "retriever_filter8005" || exit 1

# ==================== 3. 启动主训练任务 ====================
info "======================================================"
info "架构启动完毕，正式开始 GRPO LoRA 强化学习训练 (A800)！🚀"
info "======================================================"

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=28000 \
    data.max_response_length=1200 \
    +data.max_start_length=4000 \
    +data.max_obs_length=2000 \
    +data.shuffle_train_dataloader=true \
    data.filter_overlong_prompts=true \
    data.truncation='right' \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.06 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_fused_kernels=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.temperature=0.4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32000 \
    algorithm.use_kl_in_reward=false \
    +algorithm.no_think_rl=false \
    +trainer.use_critic=false \
    +trainer.do_search=true \
    trainer.critic_warmup=0 \
    trainer.logger='["wandb"]' \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=2 \
    trainer.total_training_steps=1801 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/RL_ckp/$EXPERIMENT_NAME \
    +max_turns=10 \
    ray_kwargs.ray_init.num_cpus=16 \
    +retriever.url="http://127.0.0.1:8005/retrieve" \
    +retriever.topk=8 \
    2>&1 | tee $EXPERIMENT_NAME.log