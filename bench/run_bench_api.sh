#!/bin/bash
set -e

# ================= 1. 命令行参数解析 =================
# 设置默认值
MODEL_NAME="deepseek-v3"
API_KEY=""
MAIN_API_URL=""

# 解析传入的参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --api_key) 
            API_KEY="$2"; shift ;;
        --api_url) 
            MAIN_API_URL="$2"; shift ;;
        --model) 
            MODEL_NAME="$2"; shift ;;
        *) 
            echo "未知的参数: $1"
            echo "用法: bash run_bench_api.sh --api_key <Token> --api_url <URL> [--model <模型名>]"
            exit 1 ;;
    esac
    shift
done

# 校验必填项
if [ -z "$API_KEY" ] || [ -z "$MAIN_API_URL" ]; then
    echo -e "\033[31m[ERROR] 缺少必要的参数！\033[0m"
    echo "用法: bash run_bench_api.sh --api_key <Token> --api_url <URL>"
    exit 1
fi

# 导出为环境变量，让底层的 python 脚本 (shared_agent.py) 能读到
export API_KEY
export MAIN_API_URL
export MODEL_NAME
export USE_DIRECT_API="true"

# ================= 1. 配置全局与 API 变量 =================
export BASE_DIR="${BASE_DIR:-/data/panghuaiwen/legal_R1}"
export CONDA_HOME="${CONDA_HOME:-/data/panghuaiwen/miniconda3}"



# Judge 模型配置 (如果评测打分依然用本地模型，则保留此配置)
JUDGE_MODEL_NAME="Qwen3-8B-Judge" 
WORKERS="${WORKERS:-32}"
JUDGE_VLLM_PORT=8009
PORT_TIMEOUT=1600  

# 假端口，传给 infer 脚本作占位符（由于 shared_agent 做了短路，这些不会被真正请求）
FAKE_VLLM_PORT=8007
FAKE_SUMMARY_PORT=8008

if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
    source "$CONDA_HOME/etc/profile.d/conda.sh"
fi

info() { echo -e "\033[32m[INFO]\033[0m $1"; }
error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

# ================= 2. 启动 Judge 服务 (本地打分模型) =================
# 如果你的 eval 也是调 API，这部分可以全部注释掉
kill_session() {
    if tmux has-session -t "$1" 2>/dev/null; then
        info "清理历史会话: $1"
        tmux kill-session -t "$1"
        sleep 1
    fi
}

info "启动 vLLM 裁判模型..."
kill_session "vllm_judge"
tmux new -d -s vllm_judge "export CUDA_VISIBLE_DEVICES=2; source $CONDA_HOME/etc/profile.d/conda.sh && conda activate vllm_server; python -m vllm.entrypoints.openai.api_server --model ${BASE_DIR}/model/Qwen/Qwen3-8B --served-model-name ${JUDGE_MODEL_NAME} --port ${JUDGE_VLLM_PORT} --gpu-memory-utilization 0.85 --max-model-len 22000 || sleep 86400"

info "等待 Judge 端口就绪..."
while ! curl -s -f --noproxy '*' "http://127.0.0.1:${JUDGE_VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 3
done
info "Judge 模型就绪！开始执行 API 评测！🚀"


# ================= 3. UCL Bench =================
info "================== [1/3] UCL Bench (API Mode) =================="
conda activate searchr1_new

UCL_RES_PATH="${BASE_DIR}/dataset/result/res_result/${MODEL_NAME}_ucl_eval_result.json"
UCL_SCORE_PATH="${BASE_DIR}/dataset/result/score_result/${MODEL_NAME}_ucl_score.json"
UCL_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/UCL/${MODEL_NAME}_ucl.json"
UCL_CHATGPT_REF="${BASE_DIR}/dataset/result/res_result/qwen3_8B_eval_result.json" 

# infer_脚本的参数原样传递，但在 shared_agent 内部会被短路
info " -> 1. 推理阶段"
python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_infer.py \
    --data_path "${BASE_DIR}/UCL-bench/dataset/legal_data_sample.json" \
    --result_path "${UCL_RES_PATH}" \
    --model_name "${MODEL_NAME}" \
    --vllm_url "http://127.0.0.1:${FAKE_VLLM_PORT}" \
    --summary_port ${FAKE_SUMMARY_PORT} \
    --retrieve_path "http://127.0.0.1:8005/retrieve" \
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


# ================= 4. LawBench =================
info "================== [2/3] LawBench (API Mode) =================="
conda activate lawbench

LAWBENCH_PRED_DIR="${BASE_DIR}/lawbench/test/prediction/zero_shot/${MODEL_NAME}"
LAWBENCH_SCORE_DIR="${BASE_DIR}/lawbench/test/result/${MODEL_NAME}_scored"
LAWBENCH_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/lawbench/${MODEL_NAME}_lawbench.json"

info " -> 1. 推理阶段"
python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_infer.py \
    --data_dir "${BASE_DIR}/lawbench/test/data/zero_shot" \
    --output_dir "${LAWBENCH_PRED_DIR}" \
    --model_name "${MODEL_NAME}" \
    --vllm_url "http://127.0.0.1:${FAKE_VLLM_PORT}" \
    --summary_port ${FAKE_SUMMARY_PORT} \
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


# ================= 5. LexEval =================
info "================== [3/3] LexEval (API Mode) =================="
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
                --vllm_url "http://127.0.0.1:${FAKE_VLLM_PORT}" \
                --summary_port ${FAKE_SUMMARY_PORT} \
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

info "========================================================="
info "🎉 API 评测任务圆满结束！"
info "========================================================="