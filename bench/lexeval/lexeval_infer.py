import os
import sys
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

os.umask(0)
logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from shared_agent import VLLM_Retriever_Agent, get_universal_vllm_summary

import time

def process_single_item(item, idx, agent, summary_port):
    start_time = time.time()
    query = item.get("question", item.get("input", ""))
    
    try:
        history, agent_metrics = agent.gen(query=query)
        summary, sum_p_tok, sum_c_tok = get_universal_vllm_summary(query, history, summary_port, model_name="Qwen3-8B")
    except Exception as e:
        logging.error(f"[Item {idx}] Agent Generation Failed: {e}")
        history, summary = "Error", "Error"
        agent_metrics = {"tool_latency_sec": 0, "rag_count": 0, "main_total_prompt_tokens": 0, "main_total_comp_tokens": 0, "user_prompt_tokens": 0}
        sum_p_tok, sum_c_tok = 0, 0

    total_time_sec = time.time() - start_time
    total_tokens = agent_metrics["main_total_prompt_tokens"] + agent_metrics["main_total_comp_tokens"] + sum_p_tok + sum_c_tok
    inter_agent_tokens = total_tokens - agent_metrics["user_prompt_tokens"] - sum_c_tok

    out_item = item.copy()
    out_item["thinking"] = history
    out_item["output"] = summary
    out_item["metrics"] = {
        "total_time_sec": total_time_sec,
        "tool_latency_sec": agent_metrics["tool_latency_sec"],
        "rag_count": agent_metrics["rag_count"],
        "total_tokens": total_tokens,
        "user_prompt_tokens": agent_metrics["user_prompt_tokens"],
        "completion_tokens": sum_c_tok,
        "inter_agent_tokens": inter_agent_tokens
    }
    return idx, out_item  # 返回索引以便排序

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.f_path):
        print(f"警告：文件不存在，跳过处理 - {args.f_path}")
        return
        
    logging.info(f"Start processing {args.f_path}")

    # 检查 URL 是否带路径，给出警告
    if "v1" not in args.vllm_url:
        logging.warning(f"你的 vllm_url ({args.vllm_url}) 可能缺少 /v1/completions 或 /v1/chat/completions 后缀，请检查 shared_agent.py 中是否自动拼接了路径！")

    agent = VLLM_Retriever_Agent(
        vllm_url=args.vllm_url,
        retrieve_path=args.retrieve_path,
        model_name=args.model_name,
        max_turn=args.max_turn,
        topk=args.topk
    )

    dataset = []
    with open(args.f_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            dataset = json.loads(content)
        else:
            for line in content.split('\n'):
                if line.strip():
                    dataset.append(json.loads(line.strip()))

    results = [None] * len(dataset) # 预分配列表保证顺序

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 传入 idx
        futures = {executor.submit(process_single_item, item, idx, agent, args.summary_port): idx for idx, item in enumerate(dataset)}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Inferencing {os.path.basename(args.f_path)}"):
            idx = futures[future]
            results[idx] = future.result()[1]  # 提取 out_item 按原索引就位

    base_name = os.path.basename(args.f_path) 
    name_without_ext = os.path.splitext(base_name)[0]
    outfile_name = os.path.join(args.output_dir, f"{args.model_name}_{name_without_ext}.jsonl")
    
    with open(outfile_name, 'w', encoding='utf8') as f:
        for res_dict in results:
            if res_dict is not None:
                f.write(json.dumps(res_dict, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen3-8B")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vllm_url", type=str, default="http://127.0.0.1:8007") # 重点排查！
    parser.add_argument("--summary_port", type=int, default=8008)
    parser.add_argument("--retrieve_path", type=str, default=None)
    parser.add_argument("--max_turn", type=int, default=12)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--workers", type=int, default=32)
    args, unknown = parser.parse_known_args()
    main(args)