import json
import os
import argparse
import glob
import sys
import traceback

# 1. 动态追踪父目录，跨级引用 shared_eval
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

lawbench_utils_dir = os.path.join(parent_dir, "lawbench_utils")
from shared_eval import TRADITIONAL_FUNCT_DICT

CATEGORY_MAPPING = {
    "1-1": "knowledge", "1-2": "reasoning",
    "2-1": "understanding", "2-2": "understanding", "2-3": "understanding",
    "2-4": "understanding", "2-5": "understanding", "2-6": "understanding",
    "2-7": "generation", "2-8": "reasoning", "2-9": "understanding", "2-10": "understanding",
    "3-1": "reasoning", "3-2": "reasoning", "3-3": "reasoning", "3-4": "reasoning",
    "3-5": "reasoning", "3-6": "reasoning", "3-7": "reasoning", "3-8": "consultation"
}

# 严格按照分类标准：所有“单选”“多选”的任务为客观题，其余为主观题
OBJECTIVE_TASKS = {
    "1-2", "2-2", "2-3", "2-4", "2-8", "2-9", "3-1", "3-3", "3-6"
}

# =====================================================================
# 【新增：批量处理配置项】
# 1. 在此处放入你想批量跑分数的旧轨迹文件夹路径列表（支持同时配置多个路径）
BATCH_TRAJ_DIRS = [
    "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/lawbench/test/result/deepseek-v4-flash_scored",
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/lawbench/test/result/A2Search-0516_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/lawbench/test/result/lawgpt_0519_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/lawbench/test/result/legalone-0517_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/lawbench/test/result/luwen-0517_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/lawbench/test/result/qwen3-8b-0503_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/lawbench/test/result/qwen3-8b-AR-0520_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/lawbench/test/result/qwen3-8b-AR-0521_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/lawbench/test/result/qwen3-8b-no-RAG-0519-0143_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/lawbench/test/result/qwen3-8b-SFT-AR-0521_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/lawbench/test/result/qwen3-post-train-0519_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/lawbench/test/result/R-Search-0516_scored'

]

# 2. 批量处理时，默认生成的 fix 文件的输出保存根目录
DEFAULT_OUTPUT_DIR = "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/bench_result/lawbench"
# =====================================================================


def run_single_evaluation(old_traj_dir, new_result_path):
    """封装的单个目录核心打分计算逻辑"""
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
    json_files = glob.glob(os.path.join(old_traj_dir, "*.json"))
    
    if not json_files:
        print(f"[WARN] 目录 {old_traj_dir} 中未扫描到任何 *.json 轨迹文件，跳过该项。")
        return

    for file_path in json_files:
        task_name = os.path.basename(file_path).replace(".json", "")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = data.values() if isinstance(data, dict) else data
        items_list = list(items)
        
        # === 兼容性映射 (还原 origin_prompt 防止传统评分脚本因换行符切割失败崩溃) ===
        for item in items_list:
            if "origin_prompt" not in item:
                instruction = item.get("instruction", "")
                q_text = item.get("question", "")
                if instruction.endswith("：") or instruction.endswith(":"):
                    item["origin_prompt"] = f"{instruction}\n{q_text}"
                else:
                    item["origin_prompt"] = f"{instruction}{q_text}"
                    
            if task_name == "2-1" and "origin_prompt" in item:
                if "句子：" in item["origin_prompt"] and "句子：\n" not in item["origin_prompt"]:
                    item["origin_prompt"] = item["origin_prompt"].replace("句子：", "句子：\n")
            
            if "answer" not in item and "refr" in item:
                item["answer"] = item.get("refr", "")

        count = len(items_list)
        if count == 0: continue
        
        # === 旧管道分数获取 ===
        t_llm_score = sum(item.get("eval_score", 0) for item in items_list)
        t_time = sum(item.get("metrics", {}).get("total_time_sec", 0) for item in items_list)
        t_tool = sum(item.get("metrics", {}).get("tool_latency_sec", 0) for item in items_list)
        t_rag = sum(item.get("metrics", {}).get("rag_count", 0) for item in items_list)
        t_tok = sum(item.get("metrics", {}).get("total_tokens", 0) for item in items_list)
        t_user = sum(item.get("metrics", {}).get("user_prompt_tokens", 0) for item in items_list)
        t_inter = sum(item.get("metrics", {}).get("inter_agent_tokens", 0) for item in items_list)
        t_comp = sum(item.get("metrics", {}).get("completion_tokens", 0) for item in items_list)

        # === 新管道传统分值计算 ===
        trad_score = None
        trad_abstention = None
        if task_name in TRADITIONAL_FUNCT_DICT:
            original_cwd = os.getcwd()
            try:
                os.chdir(lawbench_utils_dir) # 空间跳阅切换工作路径，确保内部依赖字典可被查阅
                score_res = TRADITIONAL_FUNCT_DICT[task_name](items_list)
                trad_score = score_res.get("score", 0) * 100.0
                if "abstention_rate" in score_res:
                    trad_abstention = score_res.get("abstention_rate", 0) * 100.0
            except Exception as e:
                print(f"\n[ERROR] 任务 {task_name} 的传统评分计算失败，详细堆栈如下:")
                traceback.print_exc()
                print("=" * 50)
            finally:
                os.chdir(original_cwd) # 强制回归原始路径
        else:
            print(f"[WARN] 找不到任务 {task_name} 的传统评分函数，传统分记为 null。")

        # === 主客观分类计算最终合并得分 (Hybrid Score) ===
        is_objective = task_name in OBJECTIVE_TASKS
        if is_objective:
            t_hybrid_score = (trad_score if trad_score is not None else 0.0) * count
        else:
            t_hybrid_score = t_llm_score

        task_category = CATEGORY_MAPPING.get(task_name, "unknown")

        task_results[task_name] = {
            "category": task_category,
            "task_type": "objective" if is_objective else "subjective",
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
                if whitespace_rate := trad_abstention:
                    cat_stats["total_trad_abs"] += (whitespace_rate * count)
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
        "bench_name": "LawBench",
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
    parser = argparse.ArgumentParser(description="主客观分类混合评估补丁脚本 - 支持硬编码批量和单文件回归模式")
    parser.add_argument('--old_traj_dir', type=str, required=False, default=None, help="旧的单任务 Eval 轨迹 JSON 所在目录")
    parser.add_argument('--new_result_path', type=str, required=False, default=None, help="输出包含新旧双评分的单文件路径")
    args = parser.parse_args()

    # 1. 检测 BATCH_TRAJ_DIRS 列表是否包含元素，若不为空则执行【批量模式】
    if BATCH_TRAJ_DIRS:
        print(f"🚀 [检测到硬编码列表] 正在启动批量合并计算，预计处理 {len(BATCH_TRAJ_DIRS)} 个目录...")
        for traj_dir in BATCH_TRAJ_DIRS:
            traj_dir = traj_dir.rstrip("/")
            if not os.path.isdir(traj_dir):
                print(f"[ERROR] 目录不存在，跳过处理: {traj_dir}")
                continue
                
            # 演绎文件名：提取最后一层文件夹名
            folder_name = os.path.basename(traj_dir)
            if "_scored" in folder_name:
                base_model_name = folder_name.split("_scored")[0]
            else:
                base_model_name = folder_name
                
            # 构建输出相对路径名加上 _fix 后缀
            target_filename = f"{base_model_name}_fix.json"
            computed_output_path = os.path.join(DEFAULT_OUTPUT_DIR, target_filename)
            
            print(f"\n⚡ 正在处理: {folder_name} -> 目标文件名: {target_filename}")
            run_single_evaluation(traj_dir, computed_output_path)
        print("\n✨ 所有批量任务均已执行完毕。")
        
    # 2. 如果 BATCH_TRAJ_DIRS 列表为空，自动回归到接收终端参数的【旧单任务模式】
    else:
        print("ℹ️ [硬编码列表为空] 正在自动回归至旧有的命令行单条处理模式...")
        if not args.old_traj_dir or not args.new_result_path:
            print("\n[ERROR] 回归旧单任务模式失败：未通过命令行检测到 --old_traj_dir 或 --new_result_path 参数！")
            print("请检查顶部列表变量 BATCH_TRAJ_DIRS 是否未正确配置，或在参数中补齐单文件路径。")
            sys.exit(1)
            
        run_single_evaluation(args.old_traj_dir, args.new_result_path)


if __name__ == "__main__":
    main()