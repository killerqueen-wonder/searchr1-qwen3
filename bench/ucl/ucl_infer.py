import os
import sys
import json
import logging
import argparse
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# # ================= 核心跨文件引用 =================
# # 1. 获取当前脚本所在目录的上一级目录 (即 legal_R1 根目录)
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # 2. 拼接出 shared_agent.py 所在的准确目录
# SHARED_DIR = os.path.join(PROJECT_ROOT, "searchr1-qwen3", "bench")

# # 3. 将该目录加入系统环境变量
# sys.path.append(SHARED_DIR)

# # 从你之前提取的公共文件中导入所需的类和函数
# try:

#     from shared_agent import VLLM_Retriever_Agent, get_universal_vllm_summary
# except ImportError:
#     print("❌ 错误：无法在根目录找到 shared_agent.py 文件，请检查文件路径！")
#     sys.exit(1)

# 加入父目录以引用 shared_agent
# 1. 获取当前脚本所在目录的绝对路径 (.../bench/ucl)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取父目录的绝对路径 (.../bench)
parent_dir = os.path.dirname(current_dir)

# 3. 将父目录加入系统环境变量
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from shared_agent import VLLM_Retriever_Agent, get_universal_vllm_summary
# =================================================

os.umask(0)
logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

def list_to_dict(data_list):
    """将处理后的平铺列表按 task_name 重新组合为字典 (UCL-Bench特有格式)"""
    classified_dict = {}
    for item in data_list:
        task_name = item.get("task_name", "unknown")
        if task_name not in classified_dict:
            classified_dict[task_name] = [item]
        else:
            classified_dict[task_name].append(item)
    return classified_dict

def process_single_item(item, agent, summary_port):
    """处理 UCL Bench 的单条数据"""
    start_time = time.time()
    
    # 1. 提取 UCL 特有的字段
    query = item["needs"]
    model_prompt = item.get("model_prompt", "")
    full_prompt = f"{model_prompt}\n{query}" if model_prompt else query
    
    # 2. 调用外部引入的 Agent 进行主推理
    history, agent_metrics = agent.gen(query=query, instruction=model_prompt)
    
    # 3. 调用外部引入的 Summary 函数进行脱水总结
    summary, sum_prompt_tok, sum_comp_tok = get_universal_vllm_summary(
        query=full_prompt, 
        history=history, 
        port=summary_port, 
        model_name="Qwen3-8B"
    )
    
    total_time_sec = time.time() - start_time
    
    # 4. 结算 Token
    user_prompt_tokens = agent_metrics["user_prompt_tokens"]
    completion_tokens = sum_comp_tok  
    total_tokens = (agent_metrics["main_total_prompt_tokens"] + agent_metrics["main_total_comp_tokens"]) + (sum_prompt_tok + sum_comp_tok)
    inter_agent_tokens = total_tokens - user_prompt_tokens - completion_tokens

    # 5. 构建符合 UCL 格式的输出
    out_item = item.copy()
    out_item["dialogue"] = f"用户：{query}\nAI助手：{summary}\n"
    out_item["model"] = agent.model_name
    out_item["thinking"] = history
    out_item["summary"] = summary
    
    # 注入性能指标
    out_item["total_time_sec"] = total_time_sec
    out_item["tool_latency_sec"] = agent_metrics["tool_latency_sec"]
    out_item["rag_count"] = agent_metrics["rag_count"]
    out_item["total_tokens"] = total_tokens
    out_item["user_prompt_tokens"] = user_prompt_tokens
    out_item["completion_tokens"] = completion_tokens
    out_item["inter_agent_tokens"] = inter_agent_tokens
    
    return out_item

def ucl_infer_main(data_path, result_path, agent, summary_port, workers=32):
    # UCL 的数据结构是 dict of lists，先拉平处理
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    dataset = [item for sublist in raw_data.values() for item in sublist]
    
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_item, item, agent, summary_port) for item in dataset]
        for future in tqdm(as_completed(futures), total=len(futures), desc="UCL-Bench Inferencing"):
            results.append(future.result())

    # 恢复为 dict 格式并保存
    final_dict = list_to_dict(results)
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default="Qwen3-8B")
    parser.add_argument('--vllm_url', type=str, default="http://127.0.0.1:8007")
    parser.add_argument('--summary_port', type=int, default=8008)
    parser.add_argument('--retrieve_path', default="http://127.0.0.1:8005/retrieve", type=str)
    parser.add_argument('--retriever', type=bool, default=True)
    parser.add_argument('--max_turn', default=12, type=int)
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--workers', default=32, type=int)
    args = parser.parse_args()

    # 初始化导入的 Agent
    agent = VLLM_Retriever_Agent(
        vllm_url=args.vllm_url,
        retrieve_path=args.retrieve_path if args.retriever else None,
        model_name=args.model_name,
        max_turn=args.max_turn,
        topk=args.topk
    )

    ucl_infer_main(args.data_path, args.result_path, agent, args.summary_port, args.workers)