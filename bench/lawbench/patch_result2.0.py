import json
import os

# 更新后的 CATEGORY_MAPPING
CATEGORY_MAPPING = {
    "1-1": "knowledge", "1-2": "reasoning",
    "2-1": "understanding", "2-2": "understanding", "2-3": "understanding",
    "2-4": "understanding", "2-5": "understanding", "2-6": "understanding",
    "2-7": "understanding", "2-8": "reasoning", "2-9": "understanding", "2-10": "understanding",
    "3-1": "reasoning", "3-2": "reasoning", "3-3": "generation", "3-4": "reasoning",
    "3-5": "reasoning", "3-6": "generation", "3-7": "reasoning", "3-8": "consultation"
}

def process_file(filepath):
    print(f"正在处理: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    task_breakdown = data.get("task_breakdown", {})
    if not task_breakdown:
        print(f"[警告] 文件 {filepath} 找不到 task_breakdown 字段，已跳过。")
        return

    # 1. 更新所有子任务的 Category
    for task_id, task_stats in task_breakdown.items():
        if task_id in CATEGORY_MAPPING:
            task_stats["category"] = CATEGORY_MAPPING[task_id]
        else:
            print(f"[警告] 发现未知的任务 ID: {task_id}")

    # 2. 重新统计 category_breakdown
    new_category_breakdown = {}
    category_totals = {}

    for task_id, task_stats in task_breakdown.items():
        cat = task_stats["category"]
        if cat not in category_totals:
            category_totals[cat] = {
                "total_hybrid_score": 0.0,
                "total_llm_score": 0.0,
                "total_trad_score": 0.0,
                "total_trad_abs": 0.0,
                "total_time": 0.0,
                "total_tool_latency": 0.0,
                "total_rag_count": 0.0,
                "total_tokens": 0.0,
                "total_user_tokens": 0.0,
                "total_inter_tokens": 0.0,
                "total_comp_tokens": 0.0,
                "sample_size": 0,
                "trad_sample_size": 0,
                "trad_abs_sample_size": 0
            }
        
        sz = task_stats.get("sample_size", 0)
        if sz == 0: continue
        
        c_totals = category_totals[cat]
        c_totals["sample_size"] += sz
        
        # 累加各项指标 (当前均分 * 样本量 = 总量)
        c_totals["total_hybrid_score"] += task_stats.get("avg_score", 0) * sz
        c_totals["total_llm_score"] += task_stats.get("llm_judge_avg_score", 0) * sz
        c_totals["total_time"] += task_stats.get("avg_time", 0) * sz
        c_totals["total_tool_latency"] += task_stats.get("avg_tool_latency", 0) * sz
        c_totals["total_rag_count"] += task_stats.get("avg_rag", 0) * sz
        c_totals["total_tokens"] += task_stats.get("avg_total_tokens", 0) * sz
        c_totals["total_user_tokens"] += task_stats.get("avg_user_tokens", 0) * sz
        c_totals["total_inter_tokens"] += task_stats.get("avg_inter_tokens", 0) * sz
        c_totals["total_comp_tokens"] += task_stats.get("avg_comp_tokens", 0) * sz
        
        # 传统评分特殊处理 (因为可能存在 null 的情况)
        if task_stats.get("traditional_avg_score") is not None:
            c_totals["total_trad_score"] += task_stats["traditional_avg_score"] * sz
            c_totals["trad_sample_size"] += sz
            
        if task_stats.get("traditional_abstention_rate") is not None:
            c_totals["total_trad_abs"] += task_stats["traditional_abstention_rate"] * sz
            c_totals["trad_abs_sample_size"] += sz

    # 计算大类微平均 (Micro-average) 并构建新的 breakdown
    for cat, totals in category_totals.items():
        sz = totals["sample_size"]
        t_sz = totals["trad_sample_size"]
        t_abs_sz = totals["trad_abs_sample_size"]
        
        if sz == 0: continue
        
        new_category_breakdown[cat] = {
            "avg_score": totals["total_hybrid_score"] / sz,
            "llm_judge_avg_score": totals["total_llm_score"] / sz,
            "traditional_avg_score": totals["total_trad_score"] / t_sz if t_sz > 0 else None,
            "traditional_abstention_rate": totals["total_trad_abs"] / t_abs_sz if t_abs_sz > 0 else None,
            "avg_time": totals["total_time"] / sz,
            "avg_tool_latency": totals["total_tool_latency"] / sz,
            "avg_rag": totals["total_rag_count"] / sz,
            "avg_total_tokens": totals["total_tokens"] / sz,
            "avg_user_tokens": totals["total_user_tokens"] / sz,
            "avg_inter_tokens": totals["total_inter_tokens"] / sz,
            "avg_comp_tokens": totals["total_comp_tokens"] / sz,
            "sample_size": sz
        }
        
    data["category_breakdown"] = new_category_breakdown
    
    # 3. 构建新文件名并保存
    base_name, ext = os.path.splitext(filepath)
    new_filepath = f"{base_name}_new_0525{ext}"
    
    with open(new_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ 文件修正完成，已保存至: {new_filepath}\n")


if __name__ == "__main__":
    # 在这里硬编码你需要处理的 JSON 文件路径列表
    # 支持相对路径或绝对路径
    FILES_TO_PROCESS = [
        # "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/bench_result/lawbench/qwen3-8b-AR-0521_lawbench_api_judge_0524.json",
        '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/bench_result/lawbench/lawgpt_0505_lawbench_api_judge_0524.json',
        '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/bench_result/lawbench/luwen-0517_lawbench_api_judge_0524.json',
        '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/bench_result/lawbench/legalone-0517_lawbench_api_judge_0524.json',
        '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/bench_result/lawbench/qwen3-8b-no-RAG-0519-0143_lawbench_api_judge_0524.json',
        '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/bench_result/lawbench/A2Search-0516_lawbench_api_judge_0524.json',
        '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/bench_result/lawbench/R-Search-0516_lawbench_api_judge_0524.json',
        '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/bench_result/lawbench/qwen3-embedding-0522-2130_lawbench_api_judge_0524.json',
        '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/bench_result/lawbench/llama-nemotron-embedding-0522_lawbench_api_judge_0524.json',
        '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/bench_result/lawbench/gpt-5.4-mini_lawbench_api_judge_0524.json',
        '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/bench_result/lawbench/deepseek-v4-flash_lawbench_api_judge_0524.json',
        
        # "另外一个文件.json", 
        # "./eval_scores/某个结果.json"
    ]
    
    for file in FILES_TO_PROCESS:
        if os.path.exists(file):
            process_file(file)
        else:
            print(f"❌ 找不到文件，已跳过: {file}")