import os
import sys
import json
import random
import re
import time
import argparse
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# ====== 全局日志与监测配置开始 ======
BASE_DIR = os.environ.get(
    "BASE_DIR", 
    "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1"
)
LOG_DIR = os.path.join(BASE_DIR, "dataset", "result", "bench_result", "log")
os.makedirs(LOG_DIR, exist_ok=True)

current_time = time.strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(LOG_DIR, f"eval_pipeline_api_{current_time}.log")

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


def call_api(prompt, model_name, api_key, api_url):
    """
    支持多 worker 并发请求 API
    加入指数退避重试机制，防止高并发时触发速率限制 (Rate Limit)
    """
    url = f"{api_url.rstrip('/')}/chat/completions"
    
    # 兼容 DeepSeek 官方或类似 OpenAI 格式的鉴权
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0 # 评测需保持一致性，温度设为0
    }
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            
            # 若触发 429 速率限制，执行退避并重试
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
            
    return ""


def compute_ucl_score(evaluation_result):
    """解析 UCL 的 [[1]] [[2]] [[3]] 格式"""
    try:
        label = re.findall(r"\[\[\s*(\d)\s*\]\]", evaluation_result.strip())
        if label:
            label = label[-1]
            if label == "1": return [10, 0]
            elif label == "2": return [0, 10]
            elif label == "3": return [5, 5]
        return [-1, -1]
    except Exception:
        return [-1, -1]


def process_single_evaluation(chatgpt_item, model_item, data_info, api_key, api_url, model_name):
    evaluation_prompt = data_info["evaluation_prompt"]
    evaluation_hints = data_info["evaluation_hints"]
    information = data_info["information"]
    needs = data_info["needs"]
    model_content = model_item.get("summary", model_item.get("thinking", ""))

    # 随机打乱对话顺序避免大模型的位置偏见 (Position Bias)
    if random.random() < 0.5:
        input_text = evaluation_prompt.format(
            information=information, needs=needs, evaluation_hints=evaluation_hints,
            dialogue1=chatgpt_item["dialogue"].strip(), dialogue2=model_content.strip()
        ).strip()
        
        evaluate_result = call_api(input_text, model_name=model_name, api_key=api_key, api_url=api_url)
        evaluate_score = compute_ucl_score(evaluate_result)
    else:
        input_text = evaluation_prompt.format(
            information=information, needs=needs, evaluation_hints=evaluation_hints,
            dialogue1=model_content.strip(), dialogue2=chatgpt_item["dialogue"].strip()
        ).strip()
        
        evaluate_result = call_api(input_text, model_name=model_name, api_key=api_key, api_url=api_url)
        score = compute_ucl_score(evaluate_result)
        evaluate_score = [score[1], score[0]] # 翻转分数，使其与原文位置匹配

    return {
        "task_name": chatgpt_item["task_name"], "id": chatgpt_item["id"],
        "evaluation_hints": evaluation_hints, "information": information, "needs": needs,
        "chatgpt_dialogue": chatgpt_item["dialogue"], "model_dialogue": model_content,
        "evaluation_result": evaluate_result, "evaluation_score": evaluate_score
    }


def main(args):
    with open(args.chatgpt_result_path, 'r', encoding='utf-8') as f: chatgpt_result = json.load(f)
    with open(args.model_result_path, 'r', encoding='utf-8') as f: model_result = json.load(f)
    with open(args.datasource_path, 'r', encoding='utf-8') as f: datasource = json.load(f)

    total_result = {task: [] for task in datasource.keys()}
    tasks_queue = []
    
    for task_name in datasource.keys():
        for c_item in chatgpt_result.get(task_name, []):
            for m_item in model_result.get(task_name, []):
                if c_item["id"] == m_item["id"]:
                    data_info = next((d for d in datasource[task_name] if d["id"] == c_item["id"]), None)
                    if data_info: tasks_queue.append((c_item, m_item, data_info))

    # 多 worker 并发请求 API
    logger.info(f"开启线程池并发，Workers: {args.workers}, API URL: {args.api_url}")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_single_evaluation, c, m, d, args.api_key, args.api_url, args.judge_model_name
            ) for c, m, d in tasks_queue
        ]
        
        for future in tqdm(as_completed(futures), total=len(tasks_queue), desc="UCL Evaluating (API)"):
            res = future.result()
            total_result[res["task_name"]].append(res)

    os.makedirs(os.path.dirname(args.result_path) or ".", exist_ok=True)
    with open(args.result_path, 'w', encoding='utf-8') as f:
        json.dump(total_result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chatgpt_result_path', type=str, required=True)
    parser.add_argument('--model_result_path', type=str, required=True)
    parser.add_argument('--datasource_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    
    # 接收 Shell 传入的 API 参数
    parser.add_argument('--api_key', type=str, required=True, help="API Key")
    parser.add_argument('--api_url', type=str, default="https://api.deepseek.com", help="API URL (含域名或特定接口端点)")
    parser.add_argument('--judge_model_name', type=str, default="deepseek-chat") 
    parser.add_argument('--workers', type=int, default=16)
    
    main(parser.parse_args())