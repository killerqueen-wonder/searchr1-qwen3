import json
import os
import argparse
import glob
import sys

# 1. 动态追踪父目录，跨级引用 shared_eval
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

from shared_eval import TRADITIONAL_FUNCT_DICT

CATEGORY_MAPPING = {
    "1-1": "knowledge", "1-2": "reasoning",
    "2-1": "understanding", "2-2": "understanding", "2-3": "understanding",
    "2-4": "understanding", "2-5": "understanding", "2-6": "understanding",
    "2-7": "generation", "2-8": "reasoning", "2-9": "understanding", "2-10": "understanding",
    "3-1": "reasoning", "3-2": "reasoning", "3-3": "reasoning", "3-4": "reasoning",
    "3-5": "reasoning", "3-6": "reasoning", "3-7": "reasoning", "3-8": "consultation"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_dir', type=str, required=True, help="含有 eval_score 的 JSON 目录")
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    # 2. 扩充全局统计，新增传统管道分数追踪
    global_stats = {
        "total_count": 0, "total_score": 0.0, 
        "total_time": 0.0, "total_tool_latency": 0.0, "total_rag_count": 0, 
        "total_tokens": 0, "total_user_tokens": 0, "total_inter_tokens": 0, "total_comp_tokens": 0,
        "total_trad_score": 0.0, "total_trad_abs": 0.0, "trad_sample_size": 0
    }
    
    # 3. 初始化大类统计容器
    category_totals = {
        cat: {
            "total_score": 0.0, "total_time": 0.0, "total_tool_latency": 0.0,
            "total_rag_count": 0, "total_tokens": 0, "total_user_tokens": 0,
            "total_inter_tokens": 0, "total_comp_tokens": 0, "sample_size": 0,
            "total_trad_score": 0.0, "total_trad_abs": 0.0, "trad_sample_size": 0
        } for cat in set(CATEGORY_MAPPING.values())
    }
    
    task_results = {}

    for file_path in glob.glob(os.path.join(args.score_dir, "*.json")):
        task_name = os.path.basename(file_path).replace(".json", "")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 将 JSON 转为 List 送入传统函数
        items = data.values() if isinstance(data, dict) else data
        items_list = list(items)
        
        count = len(items_list)
        if count == 0: continue
        
        # === 旧管道计算 (LLM) ===
        t_score = sum(item.get("eval_score", 0) for item in items_list)
        t_time = sum(item.get("metrics", {}).get("total_time_sec", 0) for item in items_list)
        t_tool = sum(item.get("metrics", {}).get("tool_latency_sec", 0) for item in items_list)
        t_rag = sum(item.get("metrics", {}).get("rag_count", 0) for item in items_list)
        t_tok = sum(item.get("metrics", {}).get("total_tokens", 0) for item in items_list)
        t_user = sum(item.get("metrics", {}).get("user_prompt_tokens", 0) for item in items_list)
        t_inter = sum(item.get("metrics", {}).get("inter_agent_tokens", 0) for item in items_list)
        t_comp = sum(item.get("metrics", {}).get("completion_tokens", 0) for item in items_list)

        # === 新管道计算 (传统分数) ===
        trad_score = None
        trad_abstention = None
        if task_name in TRADITIONAL_FUNCT_DICT:
            try:
                score_res = TRADITIONAL_FUNCT_DICT[task_name](items_list)
                trad_score = score_res.get("score", 0) * 100.0
                if "abstention_rate" in score_res:
                    trad_abstention = score_res.get("abstention_rate", 0) * 100.0
            except Exception as e:
                print(f"[ERROR] 任务 {task_name} 的传统评分计算失败: {e}")
        else:
            print(f"[WARN] 找不到任务 {task_name} 的传统评分函数，该项传统分将记为 null。")

        task_category = CATEGORY_MAPPING.get(task_name, "unknown")

        # 4. 构建 Task 层级结果
        task_results[task_name] = {
            "category": task_category,
            "avg_score": t_score / count,
            "traditional_avg_score": trad_score,              
            "traditional_abstention_rate": trad_abstention,   
            "avg_time": t_time / count,
            "avg_tool_latency": t_tool / count,
            "avg_rag": t_rag / count,
            "avg_total_tokens": t_tok / count,
            "avg_user_tokens": t_user / count,
            "avg_inter_tokens": t_inter / count,
            "avg_comp_tokens": t_comp / count,
            "sample_size": count
        }
        
        # 累加到全局
        global_stats["total_count"] += count
        global_stats["total_score"] += t_score
        global_stats["total_time"] += t_time
        global_stats["total_tool_latency"] += t_tool
        global_stats["total_rag_count"] += t_rag
        global_stats["total_tokens"] += t_tok
        global_stats["total_user_tokens"] += t_user
        global_stats["total_inter_tokens"] += t_inter
        global_stats["total_comp_tokens"] += t_comp

        if trad_score is not None:
            global_stats["total_trad_score"] += (trad_score * count)
            if trad_abstention is not None:
                global_stats["total_trad_abs"] += (trad_abstention * count)
            global_stats["trad_sample_size"] += count

        # 累加到分类
        if task_category in category_totals:
            cat_stats = category_totals[task_category]
            cat_stats["total_score"] += t_score
            cat_stats["total_time"] += t_time
            cat_stats["total_tool_latency"] += t_tool
            cat_stats["total_rag_count"] += t_rag
            cat_stats["total_tokens"] += t_tok
            cat_stats["total_user_tokens"] += t_user
            cat_stats["total_inter_tokens"] += t_inter
            cat_stats["total_comp_tokens"] += t_comp
            cat_stats["sample_size"] += count
            
            if trad_score is not None:
                cat_stats["total_trad_score"] += (trad_score * count)
                if trad_abstention is not None:
                    cat_stats["total_trad_abs"] += (trad_abstention * count)
                cat_stats["trad_sample_size"] += count

    # 5. 计算分类微平均(Micro-average)
    category_breakdown = {}
    for cat, stats in category_totals.items():
        sz = stats["sample_size"]
        t_sz = stats["trad_sample_size"]
        if sz == 0: continue
        category_breakdown[cat] = {
            "avg_score": stats["total_score"] / sz,
            "traditional_avg_score": stats["total_trad_score"] / t_sz if t_sz > 0 else None,
            "traditional_abstention_rate": stats["total_trad_abs"] / t_sz if t_sz > 0 else None,
            "avg_time": stats["total_time"] / sz,
            "avg_tool_latency": stats["total_tool_latency"] / sz,
            "avg_rag": stats["total_rag_count"] / sz,
            "avg_total_tokens": stats["total_tokens"] / sz,
            "avg_user_tokens": stats["total_user_tokens"] / sz,
            "avg_inter_tokens": stats["total_inter_tokens"] / sz,
            "avg_comp_tokens": stats["total_comp_tokens"] / sz,
            "sample_size": sz
        }

    # 6. 构建最终报告
    g_count = global_stats["total_count"] if global_stats["total_count"] > 0 else 1
    g_trad_count = global_stats["trad_sample_size"]
    
    final_report = {
        "bench_name": "LawBench",
        "global_average_score": global_stats["total_score"] / g_count,
        "traditional_global_average_score": global_stats["total_trad_score"] / g_trad_count if g_trad_count > 0 else None,
        "traditional_global_abstention_rate": global_stats["total_trad_abs"] / g_trad_count if g_trad_count > 0 else None,
        "efficiency": {
            "avg_end_to_end_time": global_stats["total_time"] / g_count,
            "avg_tool_latency": global_stats["total_tool_latency"] / g_count,
            "avg_rag_rounds": global_stats["total_rag_count"] / g_count,
            "avg_total_tokens": global_stats["total_tokens"] / g_count,
            "avg_user_tokens": global_stats["total_user_tokens"] / g_count,
            "avg_inter_tokens": global_stats["total_inter_tokens"] / g_count,
            "avg_comp_tokens": global_stats["total_comp_tokens"] / g_count
        },
        "category_breakdown": category_breakdown,
        "task_breakdown": task_results
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    print(f"LawBench 统计报告已生成: {args.output_path}")

if __name__ == "__main__":
    main()