import json
import os
import argparse
import glob
import sys
import traceback
import tempfile
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

try:
    from evaluate import Evaluator
except ImportError:
    print("[WARN] 无法从 evaluate.py 导入 Evaluator。传统评分可能无法正常工作。")
    Evaluator = None

CATEGORY_MAPPING = {
    "1_1": "understanding", "1_2": "knowledge", "1_3": "reasoning",
    "2_1": "understanding", "2_2": "understanding", "2_3": "understanding",
    "2_4": "understanding", "2_5": "understanding",
    "3_1": "reasoning", "3_2": "reasoning", "3_3": "reasoning",
    "3_4": "reasoning", "3_5": "reasoning", "3_6": "reasoning",
    "4_1": "reasoning", "4_2": "understanding",
    "5_1": "generation", "5_2": "generation", "5_3": "generation", "5_4": "generation",
    "6_1": "consultation", "6_2": "consultation", "6_3": "consultation"
}

BATCH_TRAJ_DIRS = [
    "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/LexEval/evaluation_output/deepseek-v4-flash_scored",
]

DEFAULT_OUTPUT_DIR = "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/bench_result/lexeval"

def run_single_evaluation(old_traj_dir, new_result_path):
    global_stats = {
        "total_count": 0, "total_llm_score": 0.0, "total_hybrid_score": 0.0, 
        "total_time": 0.0, "total_tool_latency": 0.0, "total_rag_count": 0, 
        "total_tokens": 0, "total_user_tokens": 0, "total_inter_tokens": 0, "total_comp_tokens": 0,
        "total_trad_score": 0.0, "total_trad_abs": 0.0, "trad_sample_size": 0
    }
    
    category_totals = {
        cat: {
            "total_llm_score": 0.0, "total_hybrid_score": 0.0,
            "total_time": 0.0, "total_tool_latency": 0.0,
            "total_rag_count": 0, "total_tokens": 0, "total_user_tokens": 0,
            "total_inter_tokens": 0, "total_comp_tokens": 0, "sample_size": 0,
            "total_trad_score": 0.0, "total_trad_abs": 0.0, "trad_sample_size": 0
        } for cat in set(CATEGORY_MAPPING.values())
    }
    
    task_results = {}
    jsonl_files = glob.glob(os.path.join(old_traj_dir, "*.jsonl"))
    
    if not jsonl_files:
        print(f"[WARN] 目录 {old_traj_dir} 中未扫描到任何 *.jsonl 轨迹文件。")
        return

    for file_path in jsonl_files:
        raw_name = os.path.basename(file_path).replace(".jsonl", "")
        
        # 【核心修复】：使用正则提取真实的 task ID（如从 deepseek-v4-flash_5_1 提取出 5_1）
        match = re.search(r'(\d+_\d+)$', raw_name)
        task_name = match.group(1) if match else raw_name
        
        items_list = []
        
        # === 【分类逻辑】：基于干净的 task_name 进行判断 ===
        is_subjective = task_name.startswith("5_")
        is_objective = not is_subjective
        
        has_null = False
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # 处理 null 答案，保护底层 evaluate.py 不崩溃，单条0分逻辑不变
                    if item.get("answer") is None:
                        item["answer"] = ""  
                        has_null = True
                    items_list.append(item)
        
        count = len(items_list)
        if count == 0: continue
        
        safe_file_path = file_path
        if has_null:
            fd, safe_file_path = tempfile.mkstemp(suffix='.jsonl', text=True)
            with os.fdopen(fd, 'w', encoding='utf-8') as tf:
                for item in items_list:
                    tf.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # === LLM 管道分数获取 ===
        t_llm_score = sum(item.get("eval_score", 0) for item in items_list)
        t_time = sum(item.get("metrics", {}).get("total_time_sec", 0) for item in items_list)
        t_tool = sum(item.get("metrics", {}).get("tool_latency_sec", 0) for item in items_list)
        t_rag = sum(item.get("metrics", {}).get("rag_count", 0) for item in items_list)
        t_tok = sum(item.get("metrics", {}).get("total_tokens", 0) for item in items_list)
        t_user = sum(item.get("metrics", {}).get("user_prompt_tokens", 0) for item in items_list)
        t_inter = sum(item.get("metrics", {}).get("inter_agent_tokens", 0) for item in items_list)
        t_comp = sum(item.get("metrics", {}).get("completion_tokens", 0) for item in items_list)

        # === 调用 evaluate.py 的传统评分 ===
        trad_score = None
        trad_abstention = None

        if Evaluator is not None:
            try:
                if is_objective:
                    evaluator = Evaluator(file_path=safe_file_path, task_type='multiple_choice', metric='Accuracy')
                else:
                    evaluator = Evaluator(file_path=safe_file_path, task_type='generation', metric='Rouge_L')
                trad_score_val = evaluator.eval()
                trad_score = trad_score_val * 100.0 # 转百分制
            except Exception as e:
                print(f"\n[ERROR] 任务 {task_name} 的传统评分计算失败:")
                traceback.print_exc()
            finally:
                if has_null and os.path.exists(safe_file_path):
                    os.remove(safe_file_path)

        # === 主客观融合计分 (Hybrid Score) ===
        if is_objective:
            t_hybrid_score = (trad_score if trad_score is not None else 0.0) * count
        else:
            t_hybrid_score = t_llm_score

        task_category = CATEGORY_MAPPING.get(task_name, "unknown")

        # 写入 task_results 时使用干净的任务名（如 "5_1"）
        task_results[task_name] = {
            "category": task_category,
            "task_type": "subjective" if is_subjective else "objective",
            "avg_score": t_hybrid_score / count,             
            "llm_judge_avg_score": t_llm_score / count,      
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
        
        global_stats["total_count"] += count
        global_stats["total_llm_score"] += t_llm_score
        global_stats["total_hybrid_score"] += t_hybrid_score
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

        if task_category in category_totals:
            cat_stats = category_totals[task_category]
            cat_stats["total_llm_score"] += t_llm_score
            cat_stats["total_hybrid_score"] += t_hybrid_score
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

    category_breakdown = {}
    for cat, stats in category_totals.items():
        sz = stats["sample_size"]
        t_sz = stats["trad_sample_size"]
        if sz == 0: continue
        category_breakdown[cat] = {
            "avg_score": stats["total_hybrid_score"] / sz,           
            "llm_judge_avg_score": stats["total_llm_score"] / sz,    
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

    g_count = global_stats["total_count"] if global_stats["total_count"] > 0 else 1
    g_trad_count = global_stats["trad_sample_size"]
    
    final_report = {
        "bench_name": "LexEval",
        "global_average_score": global_stats["total_hybrid_score"] / g_count,
        "global_llm_judge_score": global_stats["total_llm_score"] / g_count,
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

    os.makedirs(os.path.dirname(new_result_path), exist_ok=True)
    with open(new_result_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    print(f"--> [SUCCESS] 报告成功生成: {new_result_path}")

def main():
    parser = argparse.ArgumentParser(description="LexEval 主客观分类混合评估补丁脚本")
    parser.add_argument('--old_traj_dir', type=str, required=False, default=None)
    parser.add_argument('--new_result_path', type=str, required=False, default=None)
    args = parser.parse_args()

    if BATCH_TRAJ_DIRS:
        print(f"🚀 [检测到硬编码列表] 正在启动批量合并计算，预计处理 {len(BATCH_TRAJ_DIRS)} 个目录...")
        for traj_dir in BATCH_TRAJ_DIRS:
            traj_dir = traj_dir.rstrip("/")
            if not os.path.isdir(traj_dir):
                print(f"[ERROR] 目录不存在，跳过处理: {traj_dir}")
                continue
                
            folder_name = os.path.basename(traj_dir)
            base_model_name = folder_name.split("_scored")[0] if "_scored" in folder_name else folder_name
            target_filename = f"{base_model_name}_lexeval_fix.json"
            computed_output_path = os.path.join(DEFAULT_OUTPUT_DIR, target_filename)
            
            print(f"\n⚡ 正在处理: {folder_name} -> 目标文件名: {target_filename}")
            run_single_evaluation(traj_dir, computed_output_path)
        print("\n✨ 所有批量任务均已执行完毕。")
    else:
        run_single_evaluation(args.old_traj_dir, args.new_result_path)

if __name__ == "__main__":
    main()