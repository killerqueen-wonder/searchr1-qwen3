import json
import os
import argparse
import glob

# 使用精确的任务名进行大类映射 (注意：兼容代码中下划线 _ 的提取格式)
CATEGORY_MAPPING = {
    "1_1": "understanding",
    "1_2": "knowledge",
    "1_3": "reasoning",
    "2_1": "understanding",
    "2_2": "understanding",
    "2_3": "understanding",
    "2_4": "understanding",
    "2_5": "understanding",
    "3_1": "reasoning",
    "3_2": "reasoning",
    "3_3": "reasoning",
    "3_4": "reasoning",
    "3_5": "reasoning",
    "3_6": "reasoning",
    "4_1": "reasoning",
    "4_2": "understanding",
    "5_1": "generation",
    "5_2": "generation",
    "5_3": "generation",
    "5_4": "generation",
    "6_1": "consultation",
    "6_2": "consultation",
    "6_3": "consultation"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_dir', type=str, required=True, help="含有 eval_score 的 JSONL 目录")
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    # 1. 初始化全局统计字典
    global_stats = {
        "count": 0, "score": 0.0,
        "total_time": 0.0, "tool_latency": 0.0, "rag_count": 0.0,
        "total_tokens": 0.0, "user_prompt_tokens": 0.0, 
        "inter_agent_tokens": 0.0, "completion_tokens": 0.0
    }
    
    # 2. 初始化大类统计容器 (用于计算微平均)
    category_totals = {
        cat: {
            "count": 0, "score": 0.0,
            "total_time": 0.0, "tool_latency": 0.0, "rag_count": 0.0,
            "total_tokens": 0.0, "user_prompt_tokens": 0.0, 
            "inter_agent_tokens": 0.0, "completion_tokens": 0.0
        } for cat in set(CATEGORY_MAPPING.values())
    }
    
    # 按照任务（文件）级别进行统计
    task_stats = {}
    
    for file_path in glob.glob(os.path.join(args.score_dir, "*.jsonl")):
        # 提取文件名作为 task_name (例如从 Qwen3-8B_1_1.jsonl 提取 1_1)
        base_name = os.path.basename(file_path)
        task_name = base_name.split("_")[-2] + "_" + base_name.split("_")[-1].replace(".jsonl", "")
        
        # 确定所属大类
        task_category = CATEGORY_MAPPING.get(task_name, "unknown")
        if task_category not in category_totals:
            category_totals[task_category] = {
                "count": 0, "score": 0.0, "total_time": 0.0, "tool_latency": 0.0, "rag_count": 0.0,
                "total_tokens": 0.0, "user_prompt_tokens": 0.0, "inter_agent_tokens": 0.0, "completion_tokens": 0.0
            }

        if task_name not in task_stats:
            task_stats[task_name] = {
                "category": task_category,
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
                
                # 提取各项性能指标
                time_val = m.get("total_time_sec", 0)
                tool_val = m.get("tool_latency_sec", 0)
                rag_val = m.get("rag_count", 0)
                tok_val = m.get("total_tokens", 0)
                user_val = m.get("user_prompt_tokens", 0)
                inter_val = m.get("inter_agent_tokens", 0)
                comp_val = m.get("completion_tokens", 0)

                # 累加全局统计
                global_stats["count"] += 1
                global_stats["score"] += score
                global_stats["total_time"] += time_val
                global_stats["tool_latency"] += tool_val
                global_stats["rag_count"] += rag_val
                global_stats["total_tokens"] += tok_val
                global_stats["user_prompt_tokens"] += user_val
                global_stats["inter_agent_tokens"] += inter_val
                global_stats["completion_tokens"] += comp_val
                
                # 累加大类统计
                c_stats = category_totals[task_category]
                c_stats["count"] += 1
                c_stats["score"] += score
                c_stats["total_time"] += time_val
                c_stats["tool_latency"] += tool_val
                c_stats["rag_count"] += rag_val
                c_stats["total_tokens"] += tok_val
                c_stats["user_prompt_tokens"] += user_val
                c_stats["inter_agent_tokens"] += inter_val
                c_stats["completion_tokens"] += comp_val

                # 累加 Task 级统计
                t_stats = task_stats[task_name]
                t_stats["count"] += 1
                t_stats["score"] += score
                t_stats["total_time"] += time_val
                t_stats["tool_latency"] += tool_val
                t_stats["rag_count"] += rag_val
                t_stats["total_tokens"] += tok_val
                t_stats["user_prompt_tokens"] += user_val
                t_stats["inter_agent_tokens"] += inter_val
                t_stats["completion_tokens"] += comp_val

    # 3. 计算分类微平均(Micro-average)
    category_breakdown = {}
    for cat, stats in category_totals.items():
        sz = stats["count"]
        if sz == 0: continue
        category_breakdown[cat] = {
            "avg_score": stats["score"] / sz,
            "avg_time": stats["total_time"] / sz,
            "avg_tool_latency": stats["tool_latency"] / sz,
            "avg_rag": stats["rag_count"] / sz,
            "avg_total_tokens": stats["total_tokens"] / sz,
            "avg_user_tokens": stats["user_prompt_tokens"] / sz,
            "avg_inter_tokens": stats["inter_agent_tokens"] / sz,
            "avg_comp_tokens": stats["completion_tokens"] / sz,
            "sample_size": sz
        }

    # 4. 构建全局输出报告
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
        "category_breakdown": category_breakdown,
        "task_breakdown": {}
    }

    # 5. 构建任务细分报告
    for task_name, t_stats in task_stats.items():
        t_count = t_stats["count"] if t_stats["count"] > 0 else 1
        final_report["task_breakdown"][task_name] = {
            "category": t_stats["category"],
            "avg_score": t_stats["score"] / t_count,
            "avg_time": t_stats["total_time"] / t_count,
            "avg_tool_latency": t_stats["tool_latency"] / t_count,
            "avg_rag": t_stats["rag_count"] / t_count,
            "avg_total_tokens": t_stats["total_tokens"] / t_count,
            "avg_user_tokens": t_stats["user_prompt_tokens"] / t_count,
            "avg_inter_tokens": t_stats["inter_agent_tokens"] / t_count,
            "avg_comp_tokens": t_stats["completion_tokens"] / t_count,
            "sample_size": t_count
        }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    print(f"LexEval 统计报告已生成: {args.output_path}")

if __name__ == "__main__":
    main()