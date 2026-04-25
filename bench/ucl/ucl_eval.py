import os
import sys
import json
import random
import re
import time
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 引入 shared_eval
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path: sys.path.append(parent_dir)
from shared_eval import call_vllm_api

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

def process_single_evaluation(chatgpt_item, model_item, data_info, port, model_name):
    evaluation_prompt = data_info["evaluation_prompt"]
    evaluation_hints = data_info["evaluation_hints"]
    information = data_info["information"]
    needs = data_info["needs"]
    model_content = model_item.get("summary", model_item.get("thinking", ""))

    # 随机打乱对话顺序避免位置偏见
    if random.random() < 0.5:
        input_text = evaluation_prompt.format(
            information=information, needs=needs, evaluation_hints=evaluation_hints,
            dialogue1=chatgpt_item["dialogue"].strip(), dialogue2=model_content.strip()
        ).strip()
        # 动态传入 model_name
        evaluate_result = call_vllm_api(input_text, model_name=model_name, port=port)
        evaluate_score = compute_ucl_score(evaluate_result)
    else:
        input_text = evaluation_prompt.format(
            information=information, needs=needs, evaluation_hints=evaluation_hints,
            dialogue1=model_content.strip(), dialogue2=chatgpt_item["dialogue"].strip()
        ).strip()
        # 动态传入 model_name
        evaluate_result = call_vllm_api(input_text, model_name=model_name, port=port)
        score = compute_ucl_score(evaluate_result)
        evaluate_score = [score[1], score[0]] # 翻转分数

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

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 在线程池任务中追加 args.judge_model_name
        futures = [executor.submit(process_single_evaluation, c, m, d, args.judge_port, args.judge_model_name) for c, m, d in tasks_queue]
        for future in tqdm(as_completed(futures), total=len(tasks_queue), desc="UCL Evaluating"):
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
    parser.add_argument('--judge_port', type=int, default=8009)
    # 核心修改：新增打分模型名称参数
    parser.add_argument('--judge_model_name', type=str, default="Qwen3-8B") 
    parser.add_argument('--workers', type=int, default=32)
    main(parser.parse_args())