import json
import os
import argparse
import glob
import sys
import traceback
import tempfile
import re
import jsonlines
import jieba
from rouge import Rouge
import numpy as np

# 增加最大递归深度，防止 rouge-l 长文本报错
sys.setrecursionlimit(20000)

from process import find_valid_substrings, normalize_zh_answer

# =====================================================================
# 内联 Evaluator 类
# =====================================================================
class Evaluator:
    def __init__(self, file_path, task_type, metric, device='cpu', model_path=None):
        self.file_path = file_path
        if task_type == 'generation':
            self.task_type = task_type
            if metric == 'Rouge_L':
                self.metric = metric
            else:
                raise ValueError(f"No metric {metric} for generation task!")
        elif task_type == 'multiple_choice':
            self.task_type = task_type
            if metric == 'Accuracy':
                self.metric = metric
            else:
                raise ValueError(f"No metric {metric} for multiple choice task!")
        else:
            raise ValueError(f"No task type {task_type}")

    def read_jsonl(self):
        with jsonlines.open(self.file_path, "r") as r:
            for obj in r:
                yield obj
                
    def eval_accuracy(self):
        total_number = 0
        correct_number = 0
        for qa_one in self.read_jsonl():
            if qa_one.get('output', "") == "":
                ans = ""
            else:
                # 调用 process.py 中的提取逻辑
                ans = find_valid_substrings(qa_one['output'])
                
            if qa_one.get('answer', "") == ans:
                correct_number += 1
            total_number += 1
            
        if total_number == 0:
            return 0
        return correct_number / total_number

    def eval_rougel(self):
        total_number = 0
        score = 0
        rouge = Rouge()
        for qa_one in self.read_jsonl():
            out_str = qa_one.get('output', "")
            ans_str = qa_one.get('answer', "")
            
            if out_str == "":
                ans = "没有 答案"
            else:
                ans = " ".join(list(jieba.cut(normalize_zh_answer(out_str), cut_all=False)))
            
            ref = " ".join(list(jieba.cut(normalize_zh_answer(ans_str), cut_all=False)))
            
            if len(ans) == 0: ans = "没有 答案"
            if len(ref) == 0: ref = "没有 答案"
            
            try:
                s = rouge.get_scores(ans, ref)
                score += s[0]['rouge-l']['f']
            except Exception:
                score += 0.0
                
            total_number += 1
            
        if total_number == 0:
            return 0
        return score / total_number
            
    def eval(self):
        if self.metric == 'Accuracy':
            return self.eval_accuracy()
        elif self.metric == 'Rouge_L':
            return self.eval_rougel()
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")


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

def main():
    parser = argparse.ArgumentParser(description="LexEval 统计主管道脚本")
    parser.add_argument('--score_dir', type=str, required=True, help="含有 eval_score 的 JSONL 目录")
    parser.add_argument('--output_path', type=str, required=True, help="输出包含新旧双评分的 Result 文件路径")
    args = parser.parse_args()

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
    jsonl_files = glob.glob(os.path.join(args.score_dir, "*.jsonl"))
    
    if not jsonl_files:
        print(f"[WARN] 目录 {args.score_dir} 中未扫描到任何 *.jsonl 轨迹文件。")
        return

    for file_path in jsonl_files:
        raw_name = os.path.basename(file_path).replace(".jsonl", "")
        
        match = re.search(r'(\d+_\d+)$', raw_name)
        task_name = match.group(1) if match else raw_name
        
        items_list = []
        is_subjective = task_name.startswith("5_")
        is_objective = not is_subjective
        
        has_null = False
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # 保护逻辑：如果 answer 为 None，赋空字符串防崩溃
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
        
        t_llm_score = sum(item.get("eval_score", 0) for item in items_list)
        t_time = sum(item.get("metrics", {}).get("total_time_sec", 0) for item in items_list)
        t_tool = sum(item.get("metrics", {}).get("tool_latency_sec", 0) for item in items_list)
        t_rag = sum(item.get("metrics", {}).get("rag_count", 0) for item in items_list)
        t_tok = sum(item.get("metrics", {}).get("total_tokens", 0) for item in items_list)
        t_user = sum(item.get("metrics", {}).get("user_prompt_tokens", 0) for item in items_list)
        t_inter = sum(item.get("metrics", {}).get("inter_agent_tokens", 0) for item in items_list)
        t_comp = sum(item.get("metrics", {}).get("completion_tokens", 0) for item in items_list)

        trad_score = None
        trad_abstention = None

        try:
            if is_objective:
                evaluator = Evaluator(file_path=safe_file_path, task_type='multiple_choice', metric='Accuracy')
            else:
                evaluator = Evaluator(file_path=safe_file_path, task_type='generation', metric='Rouge_L')
            
            trad_score_val = evaluator.eval()
            trad_score = trad_score_val * 100.0 
        except Exception as e:
            print(f"\n[ERROR] 任务 {task_name} 的传统评分计算失败:")
            traceback.print_exc()
        finally:
            if has_null and os.path.exists(safe_file_path):
                os.remove(safe_file_path)

        # === 计分融合逻辑 ===
        # 客观题：混合分采用传统评分 (Accuracy)
        # 主观题：混合分采用 LLM 评分
        if is_objective:
            t_hybrid_score = (trad_score if trad_score is not None else 0.0) * count
        else:
            t_hybrid_score = t_llm_score

        task_category = CATEGORY_MAPPING.get(task_name, "unknown")

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
    g_trad_count = global_stats["trad_sample_size"] if global_stats["trad_sample_size"] > 0 else 1
    
    final_report = {
        "bench_name": "LexEval",
        "global_average_score": global_stats["total_hybrid_score"] / g_count,
        "global_llm_judge_score": global_stats["total_llm_score"] / g_count,
        "traditional_global_average_score": global_stats["total_trad_score"] / g_trad_count,
        "traditional_global_abstention_rate": global_stats["total_trad_abs"] / g_trad_count,
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
    print(f"\n✅ LexEval 统计报告已成功生成: {args.output_path}")

if __name__ == "__main__":
    main()