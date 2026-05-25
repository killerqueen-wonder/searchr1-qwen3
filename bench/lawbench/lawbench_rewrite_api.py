import os
import json
import glob
import time
import random
import argparse
import requests
import logging
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REWRITE_PROMPT = (
    "你是一个专业、严谨的法律AI助手。请根据下方提供的【原问题】，参考【检索资料】，针对问题进行推理，并给出最终答案。\n\n"
    "【核心规则】：\n"
    "1. 请针对原问题，给出一份具备精准证据支撑、逻辑推理清晰深刻、结论完全正确且表述专业的总结性回答。\n"
    "2. 检索资料仅供参考，你需要做出自己的判断，围绕问题做出正确推理和回答。\n\n"
    "3. 直接向用户呈现最终结果，不要包含“根据系统解析”等机器视角的客套话。\n\n"
    "【原问题】：{query}\n"
    "【检索资料】：{history}\n\n"
    "最终回答："
)

def call_api(prompt, model_name, api_key, api_url):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=120)
            if response.status_code == 429:
                time.sleep((2 ** attempt) + random.random())
                continue
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"API Error ({attempt + 1}/{max_retries}): {e}")
            time.sleep(2)
    return ""
def extract_text_segment(text: str) -> str:

    ans_idx = text.rfind('<answer>')
    if ans_idx == -1:
        return text

    prefix_text = text[:ans_idx]
    pattern = r'</[^>]{1,15}>'
    matches = list(re.finditer(pattern, prefix_text))

    if matches:

        last_match = matches[-1]
        return text[:last_match.end()]
    else:
        return prefix_text
    
def process_item(item, item_key, api_key, api_url, model_name):
    instruction = item.get("instruction", "") or ""
    question = item.get("question", "") or ""
    query = f"{instruction}\n{question}".strip()
    history = item.get("thinking", "") or ""
    history=extract_text_segment(history)
    
    prompt = REWRITE_PROMPT.format(query=query, history=history)
    new_answer = call_api(prompt, model_name, api_key, api_url)
    
    out_item = item.copy()
    out_item["prediction"] = new_answer
    return item_key, out_item

def rewrite_file(input_file, output_file, args):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_item, item, key, args.api_key, args.api_url, args.model_name) for key, item in data.items()]
        for future in tqdm(as_completed(futures), total=len(data), desc=f"Rewrite {os.path.basename(input_file)}"):
            key, out_item = future.result()
            results[key] = out_item
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--api_url', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--workers', type=int, default=64)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    for file_path in glob.glob(os.path.join(args.input_dir, "*.json")):
        output_file = os.path.join(args.output_dir, os.path.basename(file_path))
        if not os.path.exists(output_file):
            rewrite_file(file_path, output_file, args)

if __name__ == "__main__":
    main()