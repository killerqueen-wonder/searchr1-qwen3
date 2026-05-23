import os
import sys
import json
import glob
import time
import random
import argparse
import requests
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====== 全局日志配置 ======
BASE_DIR = os.environ.get("BASE_DIR", "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1")
LOG_DIR = os.path.join(BASE_DIR, "dataset", "result", "bench_result", "log")
os.makedirs(LOG_DIR, exist_ok=True)

current_time = time.strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(LOG_DIR, f"lawbench_eval_api_{current_time}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ====== 导入 Prompt 与解析工具 ======
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

def process_single_item(item, item_key, api_key, api_url, model_name):
    question = item.get("question", "")
    reference = item.get("refr", "")
    ai_response = item.get("prediction", "")
    
    prompt = GENERAL_JUDGE_PROMPT.format(question=question, reference=reference, ai_response=ai_response)
    raw_eval_text = call_api(prompt, model_name=model_name, api_key=api_key, api_url=api_url)
    score = parse_score_100(raw_eval_text)
    
    out_item = item.copy()
    out_item["eval_score"] = score
    out_item["eval_reason"] = raw_eval_text
    return item_key, out_item

def evaluate_file(input_file, output_file, api_key, api_url, model_name, workers):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = {}
    
    task_name = os.path.basename(input_file).replace(".json", "")
    OBJECTIVE_TASKS = {"1-2", "2-2", "2-3", "2-4", "2-8", "2-9", "3-1", "3-3", "3-6"}
    
    if task_name in OBJECTIVE_TASKS:
        logger.info(f"⏩ 任务 {task_name} 是客观题，跳过 API Judge 打分...")
        for key, item in data.items():
            out_item = item.copy()
            out_item["eval_score"] = 0
            out_item["eval_reason"] = "[系统拦截] 客观题，直接走传统精确匹配评测，无需大模型裁判。"
            results[key] = out_item
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_single_item, item, key, api_key, api_url, model_name) for key, item in data.items()]
            for future in tqdm(as_completed(futures), total=len(data), desc=f"Eval {os.path.basename(input_file)}"):
                key, evaluated_item = future.result()
                results[key] = evaluated_item
                
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--api_url', type=str, required=True)
    parser.add_argument('--judge_model_name', type=str, required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    for file_path in glob.glob(os.path.join(args.input_dir, "*.json")):
        evaluate_file(
            file_path, 
            os.path.join(args.output_dir, os.path.basename(file_path)), 
            args.api_key, args.api_url, args.judge_model_name, args.workers
        )