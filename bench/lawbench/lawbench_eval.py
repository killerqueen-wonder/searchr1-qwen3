import os
import sys
import json
import glob
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import time
import logging
import random

# ====== 全局日志与监测配置开始 ======
BASE_DIR = os.environ.get(
    "BASE_DIR", 
    "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1"
)
LOG_DIR = os.path.join(BASE_DIR, "dataset", "result", "bench_result", "log")
os.makedirs(LOG_DIR, exist_ok=True)

# 建议 Eval 脚本的文件名带上特定前缀以和 Infer 区分
current_time = time.strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(LOG_DIR, f"eval_pipeline_{current_time}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
# ====== 全局日志与监测配置结束 ======

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path: sys.path.append(parent_dir)
from shared_eval import GENERAL_JUDGE_PROMPT, call_vllm_api, parse_score_100

def process_single_item(item, item_key, port, model_name):
    question = item.get("question", "")
    reference = item.get("refr", "")
    ai_response = item.get("prediction", "")
    
    prompt = GENERAL_JUDGE_PROMPT.format(question=question, reference=reference, ai_response=ai_response)
    raw_eval_text = call_vllm_api(prompt, model_name=model_name, port=port)
    score = parse_score_100(raw_eval_text)
    
    out_item = item.copy()
    out_item["eval_score"] = score
    out_item["eval_reason"] = raw_eval_text
    return item_key, out_item

def evaluate_file(input_file, output_file, port, model_name, workers):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = {}
    
    # =========================================================
    # 【新增逻辑】：拦截客观题，直接赋 0 分并跳过 LLM 推理
    # =========================================================
    task_name = os.path.basename(input_file).replace(".json", "")
    OBJECTIVE_TASKS = {"1-2", "2-2", "2-3", "2-4", "2-8", "2-9", "3-1", "3-3", "3-6"}
    
    if task_name in OBJECTIVE_TASKS:
        logger.info(f"⏩ 任务 {task_name} 是客观题，跳过 LLM Judge 打分，执行秒级落盘...")
        for key, item in data.items():
            out_item = item.copy()
            out_item["eval_score"] = 0
            out_item["eval_reason"] = "[系统拦截] 客观题，直接走传统精确匹配评测，无需大模型裁判。"
            results[key] = out_item
    else:
        # =========================================================
        # 【原有逻辑】：主观题依然走多线程并发请求 LLM
        # =========================================================
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_single_item, item, key, port, model_name) for key, item in data.items()]
            for future in tqdm(as_completed(futures), total=len(data), desc=f"Eval {os.path.basename(input_file)}"):
                key, evaluated_item = future.result()
                results[key] = evaluated_item
                
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--judge_port', type=int, default=8009)
    parser.add_argument('--judge_model_name', type=str, required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for file_path in glob.glob(os.path.join(args.input_dir, "*.json")):
        evaluate_file(file_path, os.path.join(args.output_dir, os.path.basename(file_path)), args.judge_port, args.judge_model_name, args.workers)