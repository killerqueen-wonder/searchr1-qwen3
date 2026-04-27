#!/bin/bash
set -e

# ================= 1. 配置全局变量 =================
# 允许外部单行命令传参覆盖
BASE_DIR="${BASE_DIR:-/data/panghuaiwen/legal_R1}"
MODEL_PATH="/data/panghuaiwen/legal_R1/model/RL_ckp/legal_exam-ppo-qwen3-8b-RL-7.2-0420/global_step_700/actor_merge"
MODEL_NAME="${MODEL_NAME:-RL_7.2_0427}"
JUDGE_MODEL_NAME="Qwen3-8B-Judge" # 为裁判模型指定独立名称
WORKERS="${WORKERS:-32}"

# 接口与并发配置
RETRIEVE_PATH="http://127.0.0.1:8005/retrieve" 
RETRIEVER_PORT=8005
VLLM_PORT=8006
MAIN_VLLM_PORT=8007
SUMMARY_VLLM_PORT=8008
JUDGE_VLLM_PORT=8009

# 等待超时设置（秒）
PORT_TIMEOUT=1600  

# ================= 2. 环境初始化 =================
# conda 初始化脚本路径
if [ -f "/data/panghuaiwen/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="/data/panghuaiwen/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
else
    echo "错误：未找到 conda.sh，请修改脚本中的 CONDA_SH 路径"
    exit 1
fi

info() { echo -e "\033[32m[INFO]\033[0m $1"; }
error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

# ----------------- 基础设施函数 -----------------
wait_for_port() {
    local port=$1
    local timeout=$2
    local session_name=$3
    local start=$(date +%s)
    info "等待端口 $port 就绪（超时 ${timeout}s）..."
    
    while true; do

        if curl -s http://127.0.0.1:$port > /dev/null 2>&1; then
            info "端口 $port 已就绪"
            return 0
        fi

        if ! tmux has-session -t "$session_name" 2>/dev/null; then
            error "Tmux会话 $session_name 已异常退出！启动失败。"
            return 1
        fi

        local now=$(date +%s)
        if [ $((now - start)) -ge $timeout ]; then
            error "端口 $port 未在 ${timeout}s 内就绪。"
            error "请执行命令查看具体报错日志：tmux attach -t $session_name"
            return 1
        fi
        sleep 3
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

# ----------------- 环境预检 -----------------
if ! command -v tmux &> /dev/null; then
    info "未检测到 tmux，尝试安装..."
    apt update && apt install -y tmux
fi

conda config --add envs_dirs /data/panghuaiwen/legal_R1/env 2>/dev/null || true

if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    error "致命错误：NVIDIA GPU 未正常挂载或驱动损坏！容器环境异常！"
    exit 1
fi

# ================= 3. 启动后台服务 =================
info "========================================================="
info " 🚀 全量 Bench 自动化评测启动 (统一架构模式)"
info " 模型路径: ${MODEL_PATH}"
info " 模型名: ${MODEL_NAME}"
info "========================================================="

# --- 3.1 启动 RAG 检索器及依赖 vLLM ---
kill_session "retriever_filter8005"
tmux new-session -d -s retriever_filter8005 -n retriever
tmux_send_commands "retriever_filter8005" \
    "conda activate retriever_filter" \
    "export TRANSFORMERS_CACHE=/data/panghuaiwen/legal_R1/model" \
    "export HF_HUB_CACHE=/data/panghuaiwen/legal_R1/model" \
    "export TRANSFORMERS_OFFLINE=1" \
    "export HF_HUB_OFFLINE=1" \
    "cd /data/panghuaiwen/legal_R1/searchr1-qwen3" \
    "git pull origin main" \
    "bash retrieval_launch_law_text2vec.sh --port $RETRIEVER_PORT --corpus_path '/data/panghuaiwen/legal_R1/dataset/law/法律法规3.0.jsonl' --case_corpus_path '/data/panghuaiwen/legal_R1/dataset/case/lecard_court_psi.jsonl' --retriever_name hybrid_filter --dictionary_path '/data/panghuaiwen/legal_R1/dataset/dictionary/THUOCL_law.txt' --search_depth 5 --bm25_weight 15 --bm25_weight_factor 2 --bm25_k1 0.15 --bm25_b 0.35 --topk 3 --retriever_model 'shibing624/text2vec-base-chinese-paraphrase' --filter_model /data/panghuaiwen/legal_R1/model/Qwen/Qwen3-8B --vllm_url http://127.0.0.1:8006/v1/completions --gpu_ids 2 --gpu_memory_limit_per_gpu 10"
info "已触发启动 RAG 检索器 (Port: $RETRIEVER_PORT)"

kill_session "vllm"
tmux new-session -d -s vllm -n vllm
tmux_send_commands "vllm" \
    "export CUDA_VISIBLE_DEVICES=2" \
    "conda activate vllm_server" \
    "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH" \
    "python -m vllm.entrypoints.openai.api_server --model /data/panghuaiwen/legal_R1/model/Qwen/Qwen3-8B --served-model-name Qwen3-8B --port $VLLM_PORT --gpu-memory-utilization 0.35 --max-model-len 25000"
info "已触发启动 RAG 依赖的 vLLM (Port: $VLLM_PORT)"

# --- 3.2 启动主路推理、总结与评测 vLLM ---
kill_session "vllm_main"
tmux new -d -s vllm_main "export CUDA_VISIBLE_DEVICES=0; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${MODEL_PATH} --served-model-name ${MODEL_NAME} --port ${MAIN_VLLM_PORT} --gpu-memory-utilization 0.4 --max-model-len 20000"

kill_session "vllm_summary"
tmux new -d -s vllm_summary "export CUDA_VISIBLE_DEVICES=1; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${BASE_DIR}/model/Qwen/Qwen3-8B --served-model-name Qwen3-8B --port ${SUMMARY_VLLM_PORT} --gpu-memory-utilization 0.4 --max-model-len 20000"

kill_session "vllm_judge"

tmux new -d -s vllm_judge "export CUDA_VISIBLE_DEVICES=3; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${BASE_DIR}/model/Qwen/Qwen3-8B --served-model-name ${JUDGE_MODEL_NAME} --port ${JUDGE_VLLM_PORT} --gpu-memory-utilization 0.4 --max-model-len 22000"

info "等待所有服务拉起..."
sleep 20
wait_for_port $VLLM_PORT $PORT_TIMEOUT "vllm"
wait_for_port $RETRIEVER_PORT $PORT_TIMEOUT "retriever_filter8005"
wait_for_port $MAIN_VLLM_PORT $PORT_TIMEOUT "vllm_main"
wait_for_port $SUMMARY_VLLM_PORT $PORT_TIMEOUT "vllm_summary"
wait_for_port $JUDGE_VLLM_PORT $PORT_TIMEOUT "vllm_judge"
info "全部服务就绪，开始执行基准测试！🚀"


# ================= 4. UCL Bench =================
info "================== [1/3] UCL Bench =================="

# 激活主环境
source $CONDA_SH
conda activate searchr1_new

UCL_RES_PATH="${BASE_DIR}/dataset/result/res_result/${MODEL_NAME}_ucl_eval_result.json"
UCL_SCORE_PATH="${BASE_DIR}/dataset/result/score_result/${MODEL_NAME}_ucl_score.json"
UCL_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/UCL/${MODEL_NAME}_ucl.json"
UCL_CHATGPT_REF="${BASE_DIR}/dataset/result/res_result/qwen3_8B_eval_result.json" 

info " -> 1. 推理阶段"
python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_infer.py \
    --data_path "${BASE_DIR}/UCL-bench/dataset/legal_data_sample.json" \
    --result_path "${UCL_RES_PATH}" \
    --model_name "${MODEL_NAME}" \
    --vllm_url "http://127.0.0.1:${MAIN_VLLM_PORT}" \
    --summary_port ${SUMMARY_VLLM_PORT} \
    --retrieve_path "${RETRIEVE_PATH}" \
    --max_turn 12 --topk 10 --workers ${WORKERS} --retriever True

info " -> 2. 评测阶段"
python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_eval.py \
    --chatgpt_result_path "${UCL_CHATGPT_REF}" \
    --model_result_path "${UCL_RES_PATH}" \
    --datasource_path "${BASE_DIR}/UCL-bench/dataset/legal_data_sample.json" \
    --result_path "${UCL_SCORE_PATH}" \
    --judge_port ${JUDGE_VLLM_PORT} \
    --judge_model_name "${JUDGE_MODEL_NAME}" \
    --workers ${WORKERS}

info " -> 3. 统计汇总"
python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_result.py \
    --score_path "${UCL_SCORE_PATH}" \
    --inference_path "${UCL_RES_PATH}" \
    --output_path "${UCL_RESULT_PATH}"


# ================= 5. LawBench =================
info "================== [2/3] LawBench =================="

# 切换为 LawBench 的独立环境
info " -> 切换为 LawBench 环境..."
source $CONDA_SH
conda activate lawbench

LAWBENCH_PRED_DIR="${BASE_DIR}/lawbench/test/prediction/zero_shot/${MODEL_NAME}"
LAWBENCH_SCORE_DIR="${BASE_DIR}/lawbench/test/result/${MODEL_NAME}_scored"
LAWBENCH_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/lawbench/${MODEL_NAME}_lawbench.json"

info " -> 1. 推理阶段"
python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_infer.py \
    --data_dir "${BASE_DIR}/lawbench/test/data/zero_shot" \
    --output_dir "${LAWBENCH_PRED_DIR}" \
    --model_name "${MODEL_NAME}" \
    --vllm_url "http://127.0.0.1:${MAIN_VLLM_PORT}" \
    --summary_port ${SUMMARY_VLLM_PORT} \
    --retrieve_path "${RETRIEVE_PATH}" \
    --max_turn 12 --topk 10 --workers ${WORKERS} --retriever

info " -> 2. 评测阶段"
python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_eval.py \
    --input_dir "${LAWBENCH_PRED_DIR}" \
    --output_dir "${LAWBENCH_SCORE_DIR}" \
    --judge_port ${JUDGE_VLLM_PORT} \
    --judge_model_name "${JUDGE_MODEL_NAME}" \
    --workers ${WORKERS}

info " -> 3. 统计汇总"
python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_result.py \
    --score_dir "${LAWBENCH_SCORE_DIR}" \
    --output_path "${LAWBENCH_RESULT_PATH}"


# ================= 6. LexEval =================
info "================== [3/3] LexEval =================="

# 切换回主环境
info " -> 切换回 searchr1_new 环境..."
source $CONDA_SH
conda activate searchr1_new

LEXEVAL_PRED_DIR="${BASE_DIR}/LexEval/model_output/zero_shot/${MODEL_NAME}"
LEXEVAL_SCORE_DIR="${BASE_DIR}/LexEval/evaluation_output/${MODEL_NAME}_scored"
LEXEVAL_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/lexeval/${MODEL_NAME}_lexeval.json"

info " -> 1. 推理阶段"

for folder_id in {1..6}; do
    if [ "$folder_id" -eq 1 ]; then max_file=3
    elif [ "$folder_id" -eq 2 ]; then max_file=5
    elif [ "$folder_id" -eq 3 ]; then max_file=6
    elif [ "$folder_id" -eq 4 ]; then max_file=2
    elif [ "$folder_id" -eq 5 ]; then max_file=4
    elif [ "$folder_id" -eq 6 ]; then max_file=3
    fi
    for (( j=1; j<=$max_file; j++ )); do
        FILE_PATH="${BASE_DIR}/LexEval/data/${folder_id}_${j}.json"
        if [ -f "$FILE_PATH" ]; then
            python ${BASE_DIR}/searchr1-qwen3/bench/lexeval/lexeval_infer.py \
                --f_path "$FILE_PATH" \
                --model_name "${MODEL_NAME}" \
                --output_dir "${LEXEVAL_PRED_DIR}" \
                --vllm_url "http://127.0.0.1:${MAIN_VLLM_PORT}" \
                --summary_port ${SUMMARY_VLLM_PORT} \
                --retrieve_path "${RETRIEVE_PATH}" \
                --workers ${WORKERS}
        fi
    done
done

info " -> 2. 评测阶段"
python ${BASE_DIR}/searchr1-qwen3/bench/lexeval/lexeval_eval.py \
    --input_dir "${LEXEVAL_PRED_DIR}" \
    --output_dir "${LEXEVAL_SCORE_DIR}" \
    --judge_port ${JUDGE_VLLM_PORT} \
    --judge_model_name "${JUDGE_MODEL_NAME}" \
    --workers ${WORKERS}

info " -> 3. 统计汇总"
python ${BASE_DIR}/searchr1-qwen3/bench/lexeval/lexeval_result.py \
    --score_dir "${LEXEVAL_SCORE_DIR}" \
    --output_path "${LEXEVAL_RESULT_PATH}"

# ================= 7. 收尾清理 =================
info "========================================================="
info "🎉 所有评测任务圆满结束！各项报告已生成于："
info "   - UCL:      ${UCL_RESULT_PATH}"
info "   - LawBench: ${LAWBENCH_RESULT_PATH}"
info "   - LexEval:  ${LEXEVAL_RESULT_PATH}"
info "========================================================="

# 杀掉后台评测模型释放显存 (保留基础 RAG 供日常调试可不杀)
# tmux kill-session -t vllm_main || true
# tmux kill-session -t vllm_summary || true
# tmux kill-session -t vllm_judge || true
# info "清理完毕，模型推理显存已释放。"