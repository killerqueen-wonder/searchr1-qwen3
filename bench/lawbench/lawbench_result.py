import json
import os
import argparse
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_dir', type=str, required=True, help="含有 eval_score 的 JSON 目录")
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    global_stats = {
        "total_count": 0, "total_score": 0.0, "total_time": 0.0,
        "total_tool_latency": 0.0, "total_rag_count": 0, "total_tokens": 0
    }
    task_results = {}

    for file_path in glob.glob(os.path.join(args.score_dir, "*.json")):
        task_name = os.path.basename(file_path).replace(".json", "")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = len(data)
        if count == 0: continue
        
        t_score = sum(item.get("eval_score", 0) for item in data.values())
        # 从 metrics 字段提取指标
        t_time = sum(item.get("metrics", {}).get("total_time_sec", 0) for item in data.values())
        t_tool = sum(item.get("metrics", {}).get("tool_latency_sec", 0) for item in data.values())
        t_rag = sum(item.get("metrics", {}).get("rag_count", 0) for item in data.values())
        t_tok = sum(item.get("metrics", {}).get("total_tokens", 0) for item in data.values())

        task_results[task_name] = {
            "avg_score": t_score / count,
            "avg_time": t_time / count,
            "avg_rag": t_rag / count,
            "sample_size": count
        }
        
        global_stats["total_count"] += count
        global_stats["total_score"] += t_score
        global_stats["total_time"] += t_time
        global_stats["total_tool_latency"] += t_tool
        global_stats["total_rag_count"] += t_rag
        global_stats["total_tokens"] += t_tok

    final_report = {
        "bench_name": "LawBench",
        "global_average_score": global_stats["total_score"] / global_stats["total_count"],
        "efficiency": {
            "avg_end_to_end_time": global_stats["total_time"] / global_stats["total_count"],
            "avg_rag_rounds": global_stats["total_rag_count"] / global_stats["total_count"],
            "avg_tokens_per_query": global_stats["total_tokens"] / global_stats["total_count"]
        },
        "task_breakdown": task_results
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    print(f"LawBench 统计报告已生成: {args.output_path}")

if __name__ == "__main__":
    main()