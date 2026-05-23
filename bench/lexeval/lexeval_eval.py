import os
import sys
import json
import glob
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
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

def process_single_item(item, idx, port, model_name):
    question = item.get("question", item.get("input", ""))
    reference = item.get("answer", "")
    ai_response = item.get("output", "")
    
    prompt = GENERAL_JUDGE_PROMPT.format(
        question=question, 
        reference=reference, 
        ai_response=ai_response
    )
    
    try:
        raw_eval_text = call_vllm_api(prompt, model_name=model_name, port=port)
        score = parse_score_100(raw_eval_text)
    except Exception as e:
        raw_eval_text = f"Eval API Error: {e}"
        score = 0.0
    
    out_item = item.copy()
    out_item["eval_score"] = score
    out_item["eval_reason"] = raw_eval_text
    return idx, out_item

def evaluate_file(input_file, output_file, port, model_name, workers):
    dataset = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
                
    results = [None] * len(dataset)
    raw_name = os.path.basename(input_file).replace(".jsonl", "")
    
    # 提取真实的任务 ID (例如从 qwen3-embedding-0522-2130_5_3 提取 5_3)
    match = re.search(r'(\d+_\d+)$', raw_name)
    task_name = match.group(1) if match else raw_name

    # 只有 5_ 开头的任务才是主观题
    is_subjective = task_name.startswith("5_")

    if not is_subjective:
        logger.info(f"⏩ 任务 {task_name} 是客观题，跳过 LLM Judge 打分...")
        for idx, item in enumerate(dataset):
            out_item = item.copy()
            # 【核心修改 2】：拦截 null 答案
            if out_item.get("answer") is None:
                out_item["eval_score"] = 0
                out_item["eval_reason"] = "[系统拦截] 答案为 null，直接判 0 分。"
            else:
                out_item["eval_score"] = 0
                out_item["eval_reason"] = "[系统拦截] 客观题，直接走传统精确匹配评测，无需大模型裁判。"
            results[idx] = out_item
    else:
        logger.info(f"🧠 任务 {task_name} 是主观题，启动 LLM Judge 打分...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, item in enumerate(dataset):
                # 主观题同样拦截 null 答案
                if item.get("answer") is None:
                    out_item = item.copy()
                    out_item["eval_score"] = 0
                    out_item["eval_reason"] = "[系统拦截] 答案为 null，直接判 0 分。"
                    results[idx] = out_item
                else:
                    futures[executor.submit(process_single_item, item, idx, port, model_name)] = idx
                    
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Eval {os.path.basename(input_file)}"):
                idx = futures[future]
                results[idx] = future.result()[1]
                
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            if res is not None:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')
            
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    
    for file_path in jsonl_files:
        base_name = os.path.basename(file_path)
        output_file = os.path.join(args.output_dir, base_name)
        if os.path.exists(output_file):
            continue
        evaluate_file(file_path, output_file, args.judge_port, args.judge_model_name, args.workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--judge_model_name', type=str, required=True)
    parser.add_argument('--judge_port', type=int, default=8009)
    parser.add_argument('--workers', type=int, default=32)
    main(parser.parse_args())