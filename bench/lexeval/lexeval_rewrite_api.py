import os
import json
import glob
import time
import random
import argparse
import requests
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
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
    data = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
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
    """
    处理文本：
    0. 删除成对的 <think>, <THOUGHT>, <syllogism>, <search> 及其内部文本（非贪婪匹配，支持多行）
    1. 寻找最后一个 <answer>
    2. 寻找它之前的最后一个 </xx> (xx长度<=15)
    3. 根据匹配情况截取并返回对应的字符串片段
    """
    
    # ================= 第0步：数据预处理 =================
    # 使用 \1 引用第一个捕获组(即匹配到的标签名)
    # .*? 实现非贪婪匹配（匹配内部最少的文本）
    # flags=re.DOTALL 确保能够匹配标签内包含换行符的情况
    remove_pattern = r'<(think|THOUGHT|syllogism|search)>.*?</\1>'
    text = re.sub(remove_pattern, '', text, flags=re.DOTALL)


    # ================= 第1步：寻找 <answer> =================
    # 寻找最后一个 <answer> 的索引
    ans_idx = text.rfind('<answer>')
    
    # 如果没有匹配到 <answer>，直接返回全字符串
    if ans_idx == -1:
        return text
        
    # 获取最后一个 <answer> 之前的所有文本片段
    prefix_text = text[:ans_idx]
    
    
    # ================= 第2步：寻找之前的 </xx> =================
    # 使用正则：</ 开头，[^>]{1,15} 表示1到15个非 > 的字符，> 结尾
    pattern = r'</[^>]{1,15}>'
    matches = list(re.finditer(pattern, prefix_text))
    
    
    # ================= 第3步：截取结果 =================
    if matches:
        # 获取最后一个匹配项
        last_match = matches[-1]
        # 返回从开头到该 </xx> 结尾的字符串片段（包含 </xx>）
        # last_match.end() 返回的是匹配项末尾在 prefix_text 中的索引
        return text[:last_match.end()]
    else:
        # 如果匹配不到 </xx>，保留字符串开头到 <answer> 的片段
        return prefix_text
def process_item(item, idx, api_key, api_url, model_name):
    instruction = item.get("instruction", "") or ""
    input_text = item.get("input", "") or ""
    query = f"{instruction}\n{input_text}".strip()
    history = item.get("thinking", "") or ""
    history=extract_text_segment(history)
    
    prompt = REWRITE_PROMPT.format(query=query, history=history)
    new_answer = call_api(prompt, model_name, api_key, api_url)
    
    out_item = item.copy()
    out_item["output"] = new_answer
    return idx, out_item

def rewrite_file(input_file, output_file, args):
    dataset = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
                
    results = [None] * len(dataset)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_item, item, idx, args.api_key, args.api_url, args.model_name): idx for idx, item in enumerate(dataset)}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Rewrite {os.path.basename(input_file)}"):
            idx = futures[future]
            results[idx] = future.result()[1]
            
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            if res is not None:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

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
    for file_path in glob.glob(os.path.join(args.input_dir, "*.jsonl")):
        output_file = os.path.join(args.output_dir, os.path.basename(file_path))
        if not os.path.exists(output_file):
            rewrite_file(file_path, output_file, args)

if __name__ == "__main__":
    main()