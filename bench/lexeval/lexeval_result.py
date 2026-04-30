import json
import os
import argparse
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_dir', type=str, required=True, help="含有 eval_score 的 JSONL 目录")
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    # 初始化全局统计字典，包含所有 7 个指标
    global_stats = {
        "count": 0, "score": 0.0,
        "total_time": 0.0, "tool_latency": 0.0, "rag_count": 0.0,
        "total_tokens": 0.0, "user_prompt_tokens": 0.0, 
        "inter_agent_tokens": 0.0, "completion_tokens": 0.0
    }
    
    # 按照任务（文件）级别进行统计
    task_stats = {}
    
    for file_path in glob.glob(os.path.join(args.score_dir, "*.jsonl")):
        # 提取文件名作为 task_name (例如从 Qwen3-8B_1_1.jsonl 提取 1_1)
        base_name = os.path.basename(file_path)
        task_name = base_name.split("_")[-2] + "_" + base_name.split("_")[-1].replace(".jsonl", "")
        
        if task_name not in task_stats:
            task_stats[task_name] = {
                "count": 0, "score": 0.0,
                "total_time": 0.0, "tool_latency": 0.0, "rag_count": 0.0,
                "total_tokens": 0.0, "user_prompt_tokens": 0.0, 
                "inter_agent_tokens": 0.0, "completion_tokens": 0.0
            }

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                score = item.get("eval_score", 0)
                m = item.get("metrics", {})
                
                # 累加全局统计
                global_stats["count"] += 1
                global_stats["score"] += score
                global_stats["total_time"] += m.get("total_time_sec", 0)
                global_stats["tool_latency"] += m.get("tool_latency_sec", 0)
                global_stats["rag_count"] += m.get("rag_count", 0)
                global_stats["total_tokens"] += m.get("total_tokens", 0)
                global_stats["user_prompt_tokens"] += m.get("user_prompt_tokens", 0)
                global_stats["inter_agent_tokens"] += m.get("inter_agent_tokens", 0)
                global_stats["completion_tokens"] += m.get("completion_tokens", 0)
                
                # 累加 Task 级统计
                t_stats = task_stats[task_name]
                t_stats["count"] += 1
                t_stats["score"] += score
                t_stats["total_time"] += m.get("total_time_sec", 0)
                t_stats["tool_latency"] += m.get("tool_latency_sec", 0)
                t_stats["rag_count"] += m.get("rag_count", 0)
                t_stats["total_tokens"] += m.get("total_tokens", 0)
                t_stats["user_prompt_tokens"] += m.get("user_prompt_tokens", 0)
                t_stats["inter_agent_tokens"] += m.get("inter_agent_tokens", 0)
                t_stats["completion_tokens"] += m.get("completion_tokens", 0)

    # 1. 构建全局输出报告
    g_count = global_stats["count"] if global_stats["count"] > 0 else 1
    final_report = {
        "bench_name": "LexEval",
        "global_average_score": global_stats["score"] / g_count,
        "efficiency": {
            "avg_end_to_end_time": global_stats["total_time"] / g_count,
            "avg_tool_latency": global_stats["tool_latency"] / g_count,
            "avg_rag_rounds": global_stats["rag_count"] / g_count,
            "avg_total_tokens": global_stats["total_tokens"] / g_count,
            "avg_user_tokens": global_stats["user_prompt_tokens"] / g_count,
            "avg_inter_tokens": global_stats["inter_agent_tokens"] / g_count,
            "avg_comp_tokens": global_stats["completion_tokens"] / g_count
        },
        "task_breakdown": {}
    }

    # 2. 构建任务细分报告 (包含同样的7个平均量和score)
    for task_name, t_stats in task_stats.items():
        t_count = t_stats["count"] if t_stats["count"] > 0 else 1
        final_report["task_breakdown"][task_name] = {
            "avg_score": t_stats["score"] / t_count,
            "avg_time": t_stats["total_time"] / t_count,
            "avg_rag": t_stats["rag_count"] / t_count,
            "sample_size": t_stats["count"],
            "avg_tool_latency": t_stats["tool_latency"] / t_count,
            "avg_total_tokens": t_stats["total_tokens"] / t_count,
            "avg_user_tokens": t_stats["user_prompt_tokens"] / t_count,
            "avg_inter_tokens": t_stats["inter_agent_tokens"] / t_count,
            "avg_comp_tokens": t_stats["completion_tokens"] / t_count
        }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    print(f"LexEval 统计报告已生成: {args.output_path}")

if __name__ == "__main__":
    main()