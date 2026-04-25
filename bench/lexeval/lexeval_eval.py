import os
import sys
import json
import glob
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path: sys.path.append(parent_dir)
from shared_eval import GENERAL_JUDGE_PROMPT, call_vllm_api, parse_score_100

def process_single_item(item, port, model_name):
    # 适配 LexEval 的特有字段名
    question = item.get("question", item.get("input", ""))
    reference = item.get("answer", "")
    ai_response = item.get("output", "")
    
    prompt = GENERAL_JUDGE_PROMPT.format(
        question=question, 
        reference=reference, 
        ai_response=ai_response
    )
    raw_eval_text = call_vllm_api(prompt, model_name=model_name, port=port)
    score = parse_score_100(raw_eval_text)
    
    out_item = item.copy()
    out_item["eval_score"] = score
    out_item["eval_reason"] = raw_eval_text
    return out_item

def evaluate_file(input_file, output_file, port, model_name, workers):
    dataset = []
    # 逐行读取 JSONL
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): 
                dataset.append(json.loads(line.strip()))
    
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 去掉了 item_key，直接传入 item
        futures = [executor.submit(process_single_item, item, port, model_name) for item in dataset]
        for future in tqdm(as_completed(futures), total=len(dataset), desc=f"Eval {os.path.basename(input_file)}"):
            results.append(future.result())
            
    # 逐行写入 JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
            
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    # 匹配 .jsonl 后缀的文件
    jsonl_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    
    for file_path in jsonl_files:
        base_name = os.path.basename(file_path)
        output_file = os.path.join(args.output_dir, base_name)
        if os.path.exists(output_file):
            continue
        evaluate_file(file_path, output_file, args.judge_port, args.workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--judge_port', type=int, default=8009)
    parser.add_argument('--workers', type=int, default=32)
    main(parser.parse_args())