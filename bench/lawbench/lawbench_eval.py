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