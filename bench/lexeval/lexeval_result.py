import json
import os
import argparse
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_dir', type=str, required=True, help="含有 eval_score 的 JSONL 目录")
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    global_stats = {"count": 0, "score": 0.0, "time": 0.0, "rag": 0, "tok": 0}
    
    for file_path in glob.glob(os.path.join(args.score_dir, "*.jsonl")):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                global_stats["count"] += 1
                global_stats["score"] += item.get("eval_score", 0)
                m = item.get("metrics", {})
                global_stats["time"] += m.get("total_time_sec", 0)
                global_stats["rag"] += m.get("rag_count", 0)
                global_stats["tok"] += m.get("total_tokens", 0)

    final_report = {
        "bench_name": "LexEval",
        "macro_average_score": global_stats["score"] / global_stats["count"] if global_stats["count"] > 0 else 0,
        "metrics": {
            "avg_latency": global_stats["time"] / global_stats["count"],
            "avg_rag_count": global_stats["rag"] / global_stats["count"],
            "avg_tokens": global_stats["tok"] / global_stats["count"]
        }
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    print(f"LexEval 统计报告已生成: {args.output_path}")

if __name__ == "__main__":
    main()