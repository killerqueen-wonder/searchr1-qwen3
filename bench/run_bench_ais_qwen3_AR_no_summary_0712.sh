#!/bin/bash
set -euo pipefail

# LawBench no-summary ablation for qwen3-8b-AR-0521.
# It reuses saved inference records, extracts the last <answer>...</answer>
# block from each item's "thinking", writes it into "prediction", then runs
# LawBench eval/result with the same output schema as the normal pipeline.

export BASE_DIR="${BASE_DIR:-/F00120250029/lixiang_share/panghuaiwen_share/legal_R1}"
export CONDA_HOME="${CONDA_HOME:-/data/panghuaiwen/miniconda3}"
export CONDA_SH="${CONDA_SH:-/F00120250029/lixiang_share/Data/conda/etc/profile.d/conda.sh}"

export MODEL_NAME="${MODEL_NAME:-qwen3-8b-AR-0521}"
export ABLATION_SUFFIX="${ABLATION_SUFFIX:-_no_summary_0712}"

export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/F00120250029/lixiang_share/Models/Qwen3-8B}"
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen3-8B-Judge}"
export JUDGE_VLLM_PORT="${JUDGE_VLLM_PORT:-809}"
export WORKERS="${WORKERS:-32}"
export PORT_TIMEOUT="${PORT_TIMEOUT:-1600}"

# Set SKIP_JUDGE_START=true if the judge vLLM service is already running.
export SKIP_JUDGE_START="${SKIP_JUDGE_START:-false}"

SRC_PRED_DIR="${SRC_PRED_DIR:-${BASE_DIR}/lawbench/test/prediction/zero_shot/${MODEL_NAME}}"
ABLATION_MODEL_NAME="${MODEL_NAME}${ABLATION_SUFFIX}"
ABLATION_PRED_DIR="${ABLATION_PRED_DIR:-${BASE_DIR}/lawbench/test/prediction/zero_shot/${ABLATION_MODEL_NAME}}"
ABLATION_SCORE_DIR="${ABLATION_SCORE_DIR:-${BASE_DIR}/lawbench/test/result/${ABLATION_MODEL_NAME}_scored}"
ABLATION_RESULT_PATH="${ABLATION_RESULT_PATH:-${BASE_DIR}/dataset/result/bench_result/lawbench/${ABLATION_MODEL_NAME}_lawbench.json}"
LOG_DIR="${BASE_DIR}/dataset/result/bench_result/log"

mkdir -p "${LOG_DIR}"

info() { echo -e "\033[32m[INFO]\033[0m $1"; }
warn() { echo -e "\033[33m[WARN]\033[0m $1"; }
error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

wait_for_port() {
    local port=$1
    local timeout=$2
    local session_name=$3
    local start
    start=$(date +%s)
    info "Waiting for vLLM judge port ${port} (${timeout}s timeout)..."

    while true; do
        if curl -s -f --noproxy '*' "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
            info "Port ${port} is ready."
            return 0
        fi

        if [ "${SKIP_JUDGE_START}" != "true" ] && ! tmux has-session -t "${session_name}" 2>/dev/null; then
            error "Tmux session ${session_name} exited unexpectedly."
            return 1
        fi

        local now
        now=$(date +%s)
        if [ $((now - start)) -ge "${timeout}" ]; then
            error "Port ${port} was not ready within ${timeout}s."
            if [ "${SKIP_JUDGE_START}" != "true" ]; then
                error "Check logs with: tmux attach -t ${session_name}"
            fi
            return 1
        fi
        sleep 3
    done
}

kill_session() {
    if tmux has-session -t "$1" 2>/dev/null; then
        info "Cleaning old tmux session: $1"
        tmux kill-session -t "$1"
        sleep 1
    fi
}

if [ ! -f "${CONDA_SH}" ]; then
    error "Conda init script not found: ${CONDA_SH}"
    exit 1
fi

if [ ! -d "${SRC_PRED_DIR}" ]; then
    error "Source prediction directory not found: ${SRC_PRED_DIR}"
    exit 1
fi

info "========================================================="
info " LawBench no-summary ablation"
info " Source predictions: ${SRC_PRED_DIR}"
info " Ablation predictions: ${ABLATION_PRED_DIR}"
info " Score dir: ${ABLATION_SCORE_DIR}"
info " Result path: ${ABLATION_RESULT_PATH}"
info "========================================================="

if [ "${SKIP_JUDGE_START}" != "true" ]; then
    if ! command -v tmux > /dev/null 2>&1; then
        error "tmux is required to launch the judge service."
        exit 1
    fi

    if ! command -v nvidia-smi > /dev/null 2>&1 || ! nvidia-smi > /dev/null 2>&1; then
        error "NVIDIA GPU is unavailable; cannot start local judge vLLM."
        exit 1
    fi

    kill_session "vllm_judge_no_summary_0712"
    tmux new -d -s vllm_judge_no_summary_0712 \
        "export TRITON_CACHE_DIR=~/.triton/cache_vllm_judge_no_summary_0712; export CUDA_VISIBLE_DEVICES=${JUDGE_CUDA_VISIBLE_DEVICES:-2}; source ${CONDA_SH} && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${JUDGE_MODEL_PATH} --served-model-name ${JUDGE_MODEL_NAME} --port ${JUDGE_VLLM_PORT} --gpu-memory-utilization 0.85 --max-model-len 22000 || sleep 86400"
    info "Started judge vLLM on port ${JUDGE_VLLM_PORT}."
fi

wait_for_port "${JUDGE_VLLM_PORT}" "${PORT_TIMEOUT}" "vllm_judge_no_summary_0712"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT

CONVERT_HELPER="${TMP_DIR}/lawbench_make_no_summary.py"
EVAL_HELPER="${TMP_DIR}/lawbench_eval_no_summary.py"

cat > "${CONVERT_HELPER}" <<'PY'
import argparse
import glob
import json
import os
import re


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_last_answer(text):
    if not isinstance(text, str):
        return None
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    answer = matches[-1].strip()
    return answer if answer else None


def convert_file(src_path, dst_path):
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"{src_path} is expected to contain a JSON object.")

    converted = {}
    missing = 0
    for key, item in data.items():
        out_item = item.copy()
        answer = extract_last_answer(out_item.get("thinking", ""))
        if answer is None:
            out_item["prediction"] = ""
            missing += 1
        else:
            out_item["prediction"] = answer
        converted[key] = out_item

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=4, ensure_ascii=False)
    return len(converted), missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    total_items = 0
    total_missing = 0
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No *.json files found in {args.input_dir}")

    for src_path in files:
        dst_path = os.path.join(args.output_dir, os.path.basename(src_path))
        count, missing = convert_file(src_path, dst_path)
        total_items += count
        total_missing += missing
        print(f"[convert] {os.path.basename(src_path)}: {count} items, {missing} missing <answer> blocks")

    print(f"[convert] done: {len(files)} files, {total_items} items, {total_missing} missing <answer> blocks")


if __name__ == "__main__":
    main()
PY

cat > "${EVAL_HELPER}" <<'PY'
import argparse
import glob
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

BASE_DIR = os.environ.get("BASE_DIR", "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1")
LOG_DIR = os.path.join(BASE_DIR, "dataset", "result", "bench_result", "log")
os.makedirs(LOG_DIR, exist_ok=True)

current_time = time.strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(LOG_DIR, f"eval_no_summary_0712_{current_time}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

BENCH_DIR = os.path.join(BASE_DIR, "searchr1-qwen3", "bench")
if BENCH_DIR not in sys.path:
    sys.path.append(BENCH_DIR)

from shared_eval import GENERAL_JUDGE_PROMPT, call_vllm_api, parse_score_100


OBJECTIVE_TASKS = {"1-2", "2-2", "2-3", "2-4", "2-8", "2-9", "3-1", "3-3", "3-6"}
NO_ANSWER_REASON = "[no_summary_ablation] No complete <answer>...</answer> block found in thinking; score forced to 0."


def process_single_item(item, item_key, port, model_name):
    if not str(item.get("prediction", "")).strip():
        out_item = item.copy()
        out_item["eval_score"] = 0
        out_item["eval_reason"] = NO_ANSWER_REASON
        return item_key, out_item

    question = item.get("question", "")
    reference = item.get("refr", "")
    ai_response = item.get("prediction", "")

    prompt = GENERAL_JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        ai_response=ai_response,
    )
    raw_eval_text = call_vllm_api(prompt, model_name=model_name, port=port)
    score = parse_score_100(raw_eval_text)

    out_item = item.copy()
    out_item["eval_score"] = score
    out_item["eval_reason"] = raw_eval_text
    return item_key, out_item


def evaluate_file(input_file, output_file, port, model_name, workers):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_name = os.path.basename(input_file).replace(".json", "")
    results = {}
    missing_count = 0

    if task_name in OBJECTIVE_TASKS:
        logger.info("Task %s is objective; skipping LLM judge.", task_name)
        for key, item in data.items():
            out_item = item.copy()
            if not str(out_item.get("prediction", "")).strip():
                out_item["eval_score"] = 0
                out_item["eval_reason"] = NO_ANSWER_REASON
                missing_count += 1
            else:
                out_item["eval_score"] = 0
                out_item["eval_reason"] = "[system intercept] Objective task; use traditional exact-match evaluation."
            results[key] = out_item
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_single_item, item, key, port, model_name)
                for key, item in data.items()
            ]
            for future in tqdm(as_completed(futures), total=len(data), desc=f"Eval {os.path.basename(input_file)}"):
                key, evaluated_item = future.result()
                if evaluated_item.get("eval_reason") == NO_ANSWER_REASON:
                    missing_count += 1
                results[key] = evaluated_item

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logger.info("Wrote %s; forced-zero missing-answer samples: %s", output_file, missing_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--judge_port", type=int, default=809)
    parser.add_argument("--judge_model_name", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No *.json files found in {args.input_dir}")

    for file_path in files:
        output_file = os.path.join(args.output_dir, os.path.basename(file_path))
        evaluate_file(file_path, output_file, args.judge_port, args.judge_model_name, args.workers)


if __name__ == "__main__":
    main()
PY

info "Switching to LawBench environment..."
source "${CONDA_SH}"
conda activate lawbench

info "Step 1/3: extracting final <answer> blocks into no-summary prediction files..."
python "${CONVERT_HELPER}" \
    --input_dir "${SRC_PRED_DIR}" \
    --output_dir "${ABLATION_PRED_DIR}"

info "Step 2/3: running LawBench eval with missing answers forced to 0..."
python "${EVAL_HELPER}" \
    --input_dir "${ABLATION_PRED_DIR}" \
    --output_dir "${ABLATION_SCORE_DIR}" \
    --judge_port "${JUDGE_VLLM_PORT}" \
    --judge_model_name "${JUDGE_MODEL_NAME}" \
    --workers "${WORKERS}"

info "Step 3/3: aggregating LawBench result report..."
python "${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_result.py" \
    --score_dir "${ABLATION_SCORE_DIR}" \
    --output_path "${ABLATION_RESULT_PATH}"

info "========================================================="
info "LawBench no-summary ablation complete."
info "Prediction dir: ${ABLATION_PRED_DIR}"
info "Score dir: ${ABLATION_SCORE_DIR}"
info "Result path: ${ABLATION_RESULT_PATH}"
info "========================================================="
