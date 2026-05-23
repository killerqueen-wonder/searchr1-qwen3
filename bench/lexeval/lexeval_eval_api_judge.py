import os
import sys
import json
import glob
import argparse
import requests
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time
import logging

# ====== 全局日志与监测配置开始 ======
BASE_DIR = os.environ.get(
    "BASE_DIR", 
    "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1"
)
LOG_DIR = os.path.join(BASE_DIR, "dataset", "result", "bench_result", "log")
os.makedirs(LOG_DIR, exist_ok=True)

current_time = time.strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(LOG_DIR, f"lexeval_eval_api_{current_time}.log")

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
from shared_eval import GENERAL_JUDGE_PROMPT, parse_score_100

def call_api(prompt, model_name, api_key, api_url):
    """带指数退避重试机制的 API 调用"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=120)
            if response.status_code == 429:
                sleep_time = (2 ** attempt) + random.random()
                logger.warning(f"触发限流 (429)，等待 {sleep_time:.2f} 秒后重试...")
                time.sleep(sleep_time)
                continue
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"API 调用异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            time.sleep(2)
    return "-1"

def process_single_item(item, idx, api_key, api_url, model_name):
    question = item.get("question", item.get("input", ""))
    reference = item.get("answer", "")
    ai_response = item.get("output", "")
    
    prompt = GENERAL_JUDGE_PROMPT.format(
        question=question, 
        reference=reference, 
        ai_response=ai_response
    )
    
    try:
        raw_eval_text = call_api(prompt, model_name=model_name, api_key=api_key, api_url=api_url)
        score = parse_score_100(raw_eval_text)
    except Exception as e:
        raw_eval_text = f"Eval API Error: {e}"
        score = 0.0
    
    out_item = item.copy()
    out_item["eval_score"] = score
    out_item["eval_reason"] = raw_eval_text
    return idx, out_item

def evaluate_file(input_file, output_file, api_key, api_url, model_name, workers):
    dataset = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
                
    results = [None] * len(dataset)
    raw_name = os.path.basename(input_file).replace(".jsonl", "")
    
    match = re.search(r'(\d+_\d+)$', raw_name)
    task_name = match.group(1) if match else raw_name

    is_subjective = task_name.startswith("5_")

    if not is_subjective:
        logger.info(f"⏩ 任务 {task_name} 是客观题，跳过 API Judge 打分...")
        for idx, item in enumerate(dataset):
            out_item = item.copy()
            if out_item.get("answer") is None:
                out_item["eval_score"] = 0
                out_item["eval_reason"] = "[系统拦截] 答案为 null，直接判 0 分。"
            else:
                out_item["eval_score"] = 0
                out_item["eval_reason"] = "[系统拦截] 客观题，直接走传统精确匹配评测，无需大模型裁判。"
            results[idx] = out_item
    else:
        logger.info(f"🧠 任务 {task_name} 是主观题，启动 API Judge 打分...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, item in enumerate(dataset):
                if item.get("answer") is None:
                    out_item = item.copy()
                    out_item["eval_score"] = 0
                    out_item["eval_reason"] = "[系统拦截] 答案为 null，直接判 0 分。"
                    results[idx] = out_item
                else:
                    futures[executor.submit(process_single_item, item, idx, api_key, api_url, model_name)] = idx
                    
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
        evaluate_file(file_path, output_file, args.api_key, args.api_url, args.judge_model_name, args.workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--api_url', type=str, required=True)
    parser.add_argument('--judge_model_name', type=str, required=True)
    parser.add_argument('--workers', type=int, default=32)
    main(parser.parse_args())