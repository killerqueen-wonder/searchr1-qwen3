import os
import sys
import json
import argparse
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from shared_agent import VLLM_Retriever_Agent, get_universal_vllm_summary

import time

def process_single_item(data_item, idx, agent, summary_port):
    start_time = time.time()
    instruction = data_item.get("instruction", "")
    question = data_item.get("question", "")
    full_prompt = f"{instruction}\n{question}"
    
    history, agent_metrics = agent.gen(query=question, instruction=instruction)
    summary, sum_p_tok, sum_c_tok = get_universal_vllm_summary(full_prompt, history, summary_port, agent.model_name)
    
    total_time_sec = time.time() - start_time
    total_tokens = agent_metrics["main_total_prompt_tokens"] + agent_metrics["main_total_comp_tokens"] + sum_p_tok + sum_c_tok
    inter_agent_tokens = total_tokens - agent_metrics["user_prompt_tokens"] - sum_c_tok

    return {
        "origin_idx": idx,
        "prediction": summary,
        "refr": data_item.get("answer", ""),
        "thinking": history,
        "metrics": {
            "total_time_sec": total_time_sec,
            "tool_latency_sec": agent_metrics["tool_latency_sec"],
            "rag_count": agent_metrics["rag_count"],
            "total_tokens": total_tokens,
            "user_prompt_tokens": agent_metrics["user_prompt_tokens"],
            "completion_tokens": sum_c_tok,
            "inter_agent_tokens": inter_agent_tokens
        }
    }

def process_file_concurrently(data_path, output_path, agent, summary_port, max_workers=16, limit=None):
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if limit: dataset = dataset[:limit]
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {executor.submit(process_single_item, item, i, agent, summary_port): i for i, item in enumerate(dataset)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {os.path.basename(data_path)}"):
            results.append(future.result())
            
    # 按原顺序排序
    results.sort(key=lambda x: x["origin_idx"])
    
    final_dict = {}
    for i, item in enumerate(results):
        item.pop("origin_idx", None)
        final_dict[str(i)] = item
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, indent=4, ensure_ascii=False)
        print(f'save to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--vllm_url', type=str, default="http://127.0.0.1:8007") # 你启动的vLLM API
    parser.add_argument('--retrieve_path', default="http://127.0.0.1:8008/retrieve", type=str)
    parser.add_argument('--model_name', type=str, default="Qwen3-8B")
    parser.add_argument('--max_turn', default=12, type=int)
    parser.add_argument('--summary_port', type=int, default=8008)
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--limit', default=None, type=int)#测评问题数量
    parser.add_argument('--workers', default=16, type=int, help="并发线程数")
    parser.add_argument('--retriever', action='store_true')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    agent = VLLM_Retriever_Agent(
        vllm_url=args.vllm_url,
        retrieve_path=args.retrieve_path if args.retriever else None,
        model_name=args.model_name,
        max_turn=args.max_turn,
        topk=args.topk
    )

    search_path = os.path.join(args.data_dir, "*.json")
    print(f"正在搜索路径: {search_path}") # 增加这行打印
    json_files = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    for data_file in json_files:
        file_name = os.path.basename(data_file)
        output_file = os.path.join(args.output_dir, file_name)
        
        if os.path.exists(output_file):
            print(f">>> [跳过] {file_name} 已存在。")
            continue
            
        process_file_concurrently(data_file, output_file, agent, summary_port=args.summary_port, max_workers=args.workers, limit=args.limit)
        