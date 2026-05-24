#!/bin/bash
set -e

# ================= 1. 命令行参数解析 =================
# 设置 Judge API 的默认值
JUDGE_API_KEY=""
JUDGE_API_URL="https://api.deepseek.com"
JUDGE_MODEL_NAME="deepseek-v4-flash"

# 解析传入的参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --api_key) 
            JUDGE_API_KEY="$2"; shift ;;
        --api_url) 
            JUDGE_API_URL="$2"; shift ;;
        --judge_model) 
            JUDGE_MODEL_NAME="$2"; shift ;;
        *) 
            echo "未知的参数: $1"
            echo "用法: bash run_bench_ais_legalone_api_judge.sh --api_key <Token> [--api_url <URL>] [--judge_model <模型名>]"
            exit 1 ;;
    esac
    shift
done

# 校验必填项
if [ -z "$JUDGE_API_KEY" ]; then
    echo -e "\033[31m[ERROR] 缺少必要的参数 --api_key！\033[0m"
    echo "用法: bash run_bench_ais_legalone_api_judge.sh --api_key <Token> [--api_url <URL>] [--judge_model <模型名>]"
    exit 1
fi

# ================= 2. 配置全局变量 =================
export BASE_DIR="${BASE_DIR:-/F00120250029/lixiang_share/panghuaiwen_share/legal_R1}"
export CONDA_HOME="${CONDA_HOME:-/data/panghuaiwen/miniconda3}"
WORKERS=64

# ================= 3. 定义模型列表 (硬编码) =================
# 在此列表中添加你需要循环执行的所有模型名称
MODEL_LIST=(
    "qwen3-8b-AR-0521"
    # "legalone-0517"
    # "qwen3-post-train-0519"
    # 'A2Search-0516'
    # llama-nemotron-embedding-0522
    # luwen-0517
    # qwen3-8b-0503
    # qwen3-8b-AR-0520
    # qwen3-embedding-0522-2130
    # qwen3-8b-no-RAG-0519-0143
    # qwen3-8b-SFT-AR-0521
    # R-Search-0516
    # gpt-5.4-mini
    # deepseek-v4-flash
    # qwen3-8b-hybrid-0524
    # lawgpt_0517
    # lawgpt_0505


    # "添加其他模型名称"
)

# 开启强制直通模式 (极其重要：将屏蔽 RAG 与 Summary)
export USE_DIRECT_API="true"

export CONDA_SH="/F00120250029/lixiang_share/Data/conda/etc/profile.d/conda.sh"

info() { echo -e "\033[32m[INFO]\033[0m $1"; }
error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

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

info "========================================================="
info " 🚀 极速直通评测启动 (批量模式)"
info " 方案： API [裁判: ${JUDGE_MODEL_NAME}]"
info " API 地址: ${JUDGE_API_URL}"
info " 待评测模型数量: ${#MODEL_LIST[@]}"
info "========================================================="

# 激活 Conda 环境 (只需在循环外激活一次)
source $CONDA_SH
conda activate searchr1_new

# ================= 4. 循环遍历模型列表进行评测 =================
for MODEL_NAME in "${MODEL_LIST[@]}"; do
    export MODEL_NAME # 导出当前循环的模型名称供后续环境使用
    
    info "========================================================="
    info " 正在评测模型: ${MODEL_NAME}"
    info "========================================================="

    # ================= LawBench 评测模块 =================
    info "================== [1/2] LawBench =================="

    info " -> 切换为 LawBench 环境..."
    conda deactivate # 安全起见先退出当前环境
    conda activate lawbench

    LAWBENCH_PRED_DIR="${BASE_DIR}/lawbench/test/prediction/zero_shot/${MODEL_NAME}"
    LAWBENCH_REWRITE_DIR="${LAWBENCH_PRED_DIR}-api-answer"
    LAWBENCH_SCORE_DIR="${BASE_DIR}/lawbench/test/result/${MODEL_NAME}_scored_api_answer_0525"
    LAWBENCH_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/lawbench/${MODEL_NAME}_lawbench_api_answer_0525.json"

    if [ ! -d "$LAWBENCH_PRED_DIR" ]; then
        error "未找到模型 ${MODEL_NAME} 的 LawBench 预测结果目录。跳过该评测。"
    else
        info " -> 1. 预处理重写阶段 [模型: ${MODEL_NAME}]"
        # 调用 LawBench 目录下的专属重写脚本
        python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_rewrite_api.py \
            --input_dir "${LAWBENCH_PRED_DIR}" \
            --output_dir "${LAWBENCH_REWRITE_DIR}" \
            --api_key "${JUDGE_API_KEY}" \
            --api_url "${JUDGE_API_URL}" \
            --model_name "${JUDGE_MODEL_NAME}" \
            --workers ${WORKERS}

        info " -> 2. 评测阶段 (API 并发打分) [模型: ${MODEL_NAME}]"
        # 使用重写后的输出目录 LAWBENCH_REWRITE_DIR 作为打分输入
        python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_eval_api_judge.py \
            --input_dir "${LAWBENCH_REWRITE_DIR}" \
            --output_dir "${LAWBENCH_SCORE_DIR}" \
            --api_key "${JUDGE_API_KEY}" \
            --api_url "${JUDGE_API_URL}" \
            --judge_model_name "${JUDGE_MODEL_NAME}" \
            --workers ${WORKERS}

        info " -> 3. 统计汇总 [模型: ${MODEL_NAME}]"
        python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_result.py \
            --score_dir "${LAWBENCH_SCORE_DIR}" \
            --output_path "${LAWBENCH_RESULT_PATH}"
    fi

    # ================= LexEval 评测模块 =================
    info "================== [2/2] LexEval =================="

    info " -> 切换回 searchr1_new 环境..."
    conda deactivate
    conda activate searchr1_new

    LEXEVAL_PRED_DIR="${BASE_DIR}/LexEval/model_output/zero_shot/${MODEL_NAME}"
    LEXEVAL_REWRITE_DIR="${LEXEVAL_PRED_DIR}-api-answer"
    LEXEVAL_SCORE_DIR="${BASE_DIR}/LexEval/evaluation_output/${MODEL_NAME}_scored_api_answer_0525"
    LEXEVAL_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/lexeval/${MODEL_NAME}_lexeval_api_answer_0525.json"

    if [ ! -d "$LEXEVAL_PRED_DIR" ]; then
        error "未找到模型 ${MODEL_NAME} 的 LexEval 预测结果目录。跳过该评测。"
    else
        info " -> 1. 预处理重写阶段 [模型: ${MODEL_NAME}]"
        # 调用 LexEval 目录下的专属重写脚本
        python ${BASE_DIR}/searchr1-qwen3/bench/lexeval/lexeval_rewrite_api.py \
            --input_dir "${LEXEVAL_PRED_DIR}" \
            --output_dir "${LEXEVAL_REWRITE_DIR}" \
            --api_key "${JUDGE_API_KEY}" \
            --api_url "${JUDGE_API_URL}" \
            --model_name "${JUDGE_MODEL_NAME}" \
            --workers ${WORKERS}

        info " -> 2. 评测阶段 (API 并发打分) [模型: ${MODEL_NAME}]"
        # 使用重写后的输出目录 LEXEVAL_REWRITE_DIR 作为打分输入
        python ${BASE_DIR}/searchr1-qwen3/bench/lexeval/lexeval_eval_api_judge.py \
            --input_dir "${LEXEVAL_REWRITE_DIR}" \
            --output_dir "${LEXEVAL_SCORE_DIR}" \
            --api_key "${JUDGE_API_KEY}" \
            --api_url "${JUDGE_API_URL}" \
            --judge_model_name "${JUDGE_MODEL_NAME}" \
            --workers ${WORKERS}

        info " -> 3. 统计汇总 [模型: ${MODEL_NAME}]"
        python ${BASE_DIR}/searchr1-qwen3/bench/lexeval/lexeval_result.py \
            --score_dir "${LEXEVAL_SCORE_DIR}" \
            --output_path "${LEXEVAL_RESULT_PATH}"
    fi

    # 收尾与报告打印
    info "---------------------------------------------------------"
    info "🎉 模型 ${MODEL_NAME} 评测任务结束！报告生成于："
    info "   - UCL:      ${UCL_RESULT_PATH}"
    info "   - LawBench: ${LAWBENCH_RESULT_PATH}"
    info "   - LexEval:  ${LEXEVAL_RESULT_PATH}"
    info "---------------------------------------------------------"
    echo ""

done

info "========================================================="
info " ✅ 所有模型 (${#MODEL_LIST[@]} 个) 的评测任务均已执行完毕！"
info "========================================================="