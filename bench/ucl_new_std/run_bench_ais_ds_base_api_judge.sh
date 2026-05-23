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
    # "qwen3-8b-AR-0521"
    # "legalone-0517"
    # "qwen3-post-train-0519"
    # 'A2Search-0516'
    # lawgpt_0519
    # llama-nemotron-embedding-0522
    # luwen-0517
    qwen3-8b-0503
    qwen3-8b-AR-0520
    qwen3-embedding-0522-2130
    qwen3-8b-no-RAG-0519-0143
    qwen3-8b-SFT-AR-0521
    R-Search-0516
    gpt-5.4-mini

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

    # 针对当前模型动态生成路径
    UCL_RES_PATH="${BASE_DIR}/dataset/result/res_result/${MODEL_NAME}_ucl_eval_result.json"
    UCL_SCORE_PATH="${BASE_DIR}/dataset/result/score_result/${MODEL_NAME}_ucl_score_api_judge_0524.json"
    UCL_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/UCL/${MODEL_NAME}_ucl_api_judge_0524.json"
    UCL_CHATGPT_REF="/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/res_result/deepseek_eval_result.json" 

    info " -> 2. 评测阶段 (API 并发打分) [模型: ${MODEL_NAME}]"
    python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_eval_api_judge.py \
        --chatgpt_result_path "${UCL_CHATGPT_REF}" \
        --model_result_path "${UCL_RES_PATH}" \
        --datasource_path "${BASE_DIR}/UCL-bench/dataset/legal_data_sample.json" \
        --result_path "${UCL_SCORE_PATH}" \
        --judge_model_name "${JUDGE_MODEL_NAME}" \
        --api_key "${JUDGE_API_KEY}" \
        --api_url "${JUDGE_API_URL}" \
        --workers ${WORKERS}

    info " -> 3. 统计汇总 [模型: ${MODEL_NAME}]"
    python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_result.py \
        --score_path "${UCL_SCORE_PATH}" \
        --inference_path "${UCL_RES_PATH}" \
        --output_path "${UCL_RESULT_PATH}"

    # 收尾与报告打印
    info "---------------------------------------------------------"
    info "🎉 模型 ${MODEL_NAME} 评测任务圆满结束！报告已生成于："
    info "   - UCL:      ${UCL_RESULT_PATH}"
    info "---------------------------------------------------------"
    echo "" # 打印空行，方便阅读日志
done

info "========================================================="
info " ✅ 所有模型 (${#MODEL_LIST[@]} 个) 的评测任务均已执行完毕！"
info "========================================================="