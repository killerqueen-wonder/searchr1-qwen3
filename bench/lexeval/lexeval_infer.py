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

def process_single_item(item, agent, summary_port):
    start_time = time.time()
    query = item.get("question", item.get("input", ""))
    
    history, agent_metrics = agent.gen(query=query)
    summary, sum_p_tok, sum_c_tok = get_universal_vllm_summary(query, history, summary_port, model_name="Qwen3-8B")
    
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
    return out_item

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    # 检查输入文件是否存在
    if not os.path.exists(args.f_path):
        print(f"警告：文件不存在，跳过处理 - {args.f_path}")
        return  # 或 sys.exit(0) 非零退出码表示异常
        
    task_num = args.f_path.split("/")[-1].split(".")[0].split("_")[0]
    logging.info(f"Start processing {args.f_path}")

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
            # 格式为标准 JSON List: [{"id":1}, {"id":2}]
            dataset = json.loads(content)
        else:
            # 格式为 JSONL 或以换行符分隔的 JSON
            for line in content.split('\n'):
                if line.strip():
                    dataset.append(json.loads(line.strip()))

    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_single_item, item, agent, args.summary_port) for item in dataset]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Inferencing {args.f_path.split('/')[-1]}"):
            results.append(future.result())
    base_name = os.path.basename(args.f_path) 
    name_without_ext = os.path.splitext(base_name)[0]
    outfile_name = os.path.join(args.output_dir, f"{args.model_name}_{name_without_ext}.jsonl")
    
    with open(outfile_name, 'w', encoding='utf8') as f:
        for res_dict in results:
            f.write(json.dumps(res_dict, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen3-8B")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vllm_url", type=str, default="http://127.0.0.1:8007")
    parser.add_argument("--summary_port", type=int, default=8008)
    parser.add_argument("--retrieve_path", type=str, default=None)
    parser.add_argument("--max_turn", type=int, default=12)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--workers", type=int, default=32)
    
    # 兼容原脚本传进来的冗余参数
    # parser.add_argument("--is_few_shot", action="store_true")
    # parser.add_argument("--is_vllm", action="store_true")
    # parser.add_argument("--few_shot_path", type=str)
    # parser.add_argument("--model_path", type=str)
    # parser.add_argument("--model_path_base", type=str)
    # parser.add_argument("--api_base", type=str)
    # parser.add_argument("--api_key", type=str)
    # parser.add_argument("--log_name", type=str, default='running.log')
    # parser.add_argument("--batch_size", type=int)
    # parser.add_argument("--device", type=str)
    # parser.add_argument("--tensor_parallel_size", type=int)
    # parser.add_argument("--gpu_memory_utilization", type=float)
    
    args = parser.parse_args()
    main(args)