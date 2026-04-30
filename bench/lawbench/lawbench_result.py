import json
import os
import argparse
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_dir', type=str, required=True, help="含有 eval_score 的 JSON 目录")
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    # 1. 扩充全局统计变量字典，补齐 7 个变量
    global_stats = {
        "total_count": 0, "total_score": 0.0, 
        "total_time": 0.0, "total_tool_latency": 0.0, "total_rag_count": 0, 
        "total_tokens": 0, "total_user_tokens": 0, "total_inter_tokens": 0, "total_comp_tokens": 0
    }
    task_results = {}

    for file_path in glob.glob(os.path.join(args.score_dir, "*.json")):
        task_name = os.path.basename(file_path).replace(".json", "")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 兼容处理：如果最外层是列表，直接迭代；如果是字典，迭代 values()
        items = data.values() if isinstance(data, dict) else data
        
        count = len(items)
        if count == 0: continue
        
        t_score = sum(item.get("eval_score", 0) for item in items)
        
        # 2. 从 metrics 字段提取全部 7 个指标
        t_time = sum(item.get("metrics", {}).get("total_time_sec", 0) for item in items)
        t_tool = sum(item.get("metrics", {}).get("tool_latency_sec", 0) for item in items)
        t_rag = sum(item.get("metrics", {}).get("rag_count", 0) for item in items)
        t_tok = sum(item.get("metrics", {}).get("total_tokens", 0) for item in items)
        t_user = sum(item.get("metrics", {}).get("user_prompt_tokens", 0) for item in items)
        t_inter = sum(item.get("metrics", {}).get("inter_agent_tokens", 0) for item in items)
        t_comp = sum(item.get("metrics", {}).get("completion_tokens", 0) for item in items)

        # 3. 构建 Task 层级的返回字典，包含平均分、样本量及 7 个效率均值
        task_results[task_name] = {
            "avg_score": t_score / count,
            "avg_time": t_time / count,
            "avg_tool_latency": t_tool / count,
            "avg_rag": t_rag / count,
            "avg_total_tokens": t_tok / count,
            "avg_user_tokens": t_user / count,
            "avg_inter_tokens": t_inter / count,
            "avg_comp_tokens": t_comp / count,
            "sample_size": count
        }
        
        # 累加到全局统计
        global_stats["total_count"] += count
        global_stats["total_score"] += t_score
        global_stats["total_time"] += t_time
        global_stats["total_tool_latency"] += t_tool
        global_stats["total_rag_count"] += t_rag
        global_stats["total_tokens"] += t_tok
        global_stats["total_user_tokens"] += t_user
        global_stats["total_inter_tokens"] += t_inter
        global_stats["total_comp_tokens"] += t_comp

    # 4. 构建最终报告，使用 efficiency 字段并展示 7 个平均量
    g_count = global_stats["total_count"] if global_stats["total_count"] > 0 else 1
    final_report = {
        "bench_name": "LawBench",
        "global_average_score": global_stats["total_score"] / g_count,
        "efficiency": {
            "avg_end_to_end_time": global_stats["total_time"] / g_count,
            "avg_tool_latency": global_stats["total_tool_latency"] / g_count,
            "avg_rag_rounds": global_stats["total_rag_count"] / g_count,
            "avg_total_tokens": global_stats["total_tokens"] / g_count,
            "avg_user_tokens": global_stats["total_user_tokens"] / g_count,
            "avg_inter_tokens": global_stats["total_inter_tokens"] / g_count,
            "avg_comp_tokens": global_stats["total_comp_tokens"] / g_count
        },
        "task_breakdown": task_results
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    print(f"LawBench 统计报告已生成: {args.output_path}")

if __name__ == "__main__":
    main()