import json
import os
import argparse
import glob
import sys
import traceback
import pandas as pd
import numpy as np
import jsonlines
import torch
import torch.nn as nn
import pandas as pd
import bert_score
import argparse
import jieba
from transformers import BertTokenizer, BartForConditionalGeneration
from typing import List
from rouge import Rouge
from process import BARTScorer, find_valid_substrings, normalize_zh_answer
from tqdm import tqdm

import numpy as np

# Dynamically track parent directory to cross-reference evaluate
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

# Import the Evaluator from evaluate.py for traditional scoring
# try:
#     from evaluate import Evaluator
# except ImportError:
#     print("[WARN] 无法从 evaluate.py 导入 Evaluator。传统评分可能无法正常工作。")
#     Evaluator = None

class Evaluator:
    '''
    Evaluating the score for a given task on a given metric from one model's output
    '''
    def __init__(self, file_path, task_type, metric, device='cpu', model_path=None):
        '''
        Args:
            file_path: Input file path for the model's output
            task_type: generation or multiple_choice task
            metric: metrics for evaluation, f1 or accuracy for multiple choice and rouge-l, bertscore or bartscore for generation task
            device: Using cuda or cpu to do evaluation
            model_path: path for bert model or bart model, only useful if using bertscore or bartscore to evaluate
        '''
        self.file_path = file_path
        if task_type == 'generation':
            self.task_type = task_type
            if metric == 'Rouge_L':
                self.metric = metric
            elif metric == 'Bertscore':
                self.metric = metric
                if model_path != None:
                    self.model_path = model_path
                else:
                    raise ValueError(f"Lacking bert model for evaluation")
            elif metric == 'Bartscore':
                self.metric = metric
                if model_path != None:
                    self.model_path = model_path
                else:
                    raise ValueError(f"Lacking bart model for evaluation")
            else:
                raise ValueError(f"Wrong metric for generation evaluation. It has to be 'Rouge_L', 'Bertscore' or 'Bartscore' but get {metric}")
        elif task_type == 'multiple_choice':
            self.task_type = task_type
            if metric == 'Accuracy' or metric == 'F1':
                self.metric = metric
            else:
                raise ValueError(f"Wrong metric for multiple choice evaluation. It has to be 'Accuracy' or 'F1' but get {metric}")
        else:
            raise ValueError(f"Wrong task type for evaluation. It has to be 'generation' or 'multiple_choice', but get {task_type}")
        self.device = device
        
    def eval_accuracy(self):
        '''
        Output the accuracy for the given file
        '''
        score = 0
        num = 0
        with jsonlines.open(self.file_path) as f:
            for qa_one in f:
                pred = find_valid_substrings(qa_one['output'])
                if pred == qa_one['answer']:
                    score += 1
                num += 1
        acc = score / num
        return acc
    
    def eval_f1(self):
        '''
        Output the f1-score for the given file, refers to lawbench
        '''    
        with jsonlines.open(self.file_path) as f:
            score = []
            for qa_one in f:
                pred = find_valid_substrings(qa_one['output'])
                pred_set = set(pred)
                gt_set = set(qa_one['answer'])
                precision = len(pred_set.intersection(gt_set)) / len(pred_set) if len(pred_set) > 0 else 0
                recall = len(pred_set.intersection(gt_set)) / len(gt_set) if len(gt_set) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                score.append(f1)
            f1 = sum(score) / len(score)   
        return f1
    
    def eval_rougel(self):
        '''
        Output the Rouge-L score for the given file
        '''
        with jsonlines.open(self.file_path) as f:
            score = []
            for qa_one in f:
                pred = " ".join(list(jieba.cut(normalize_zh_answer(qa_one['output']), cut_all=False)))
                ans = " ".join(list(jieba.cut(normalize_zh_answer(qa_one['answer']), cut_all=False)))
                rouge = Rouge()
                try:
                    score.append(rouge.get_scores([pred], [ans], avg=True)["rouge-l"]["f"])
                except:
                    score.append(0.0)
        rouge_l = sum(score) / len(score)
        return rouge_l
    
    def eval_bertscore(self, batch_size=10):
        '''
        Output the bertscore for the given file
        '''
        with jsonlines.open(self.file_path) as f:
            all_qa = [qa_one for qa_one in f]
        all_pred, all_gt = [qa_one['output'] for qa_one in all_qa], [qa_one['answer'] for qa_one in all_qa]
        assert (len(all_pred) == len(all_gt))
        score_p, score_r, score_f1 = bert_score.score(all_pred, all_gt, lang='zh', verbose=False, model_type=self.model_path, num_layers=8, device=self.device, batch_size=batch_size)
        bertscore = (sum(score_f1) / len(score_f1)).item()
        return bertscore
    
    def eval_bartscore(self, batch_size=10):
        '''
        Output the bartscore for the given file
        '''
        with jsonlines.open(self.file_path) as f:
            all_qa = [qa_one for qa_one in f]
        all_pred, all_gt = [qa_one['output'] for qa_one in all_qa], [qa_one['answer'] for qa_one in all_qa]
        bart_calculator = BARTScorer(checkpoint=self.model_path)
        score = bart_calculator.score(all_pred, all_gt, batch_size=batch_size)
        bartscore = sum(score) / len(score)
        return bartscore
    
    def eval(self):
        '''
        Output the evaluation result for the given file on the given metric
        '''
        if self.task_type == 'generation':
            if self.metric == 'Rouge_L':
                return self.eval_rougel()
            elif self.metric == 'Bertscore':
                return self.eval_bertscore()
            elif self.metric == 'Bartscore':
                return self.eval_bartscore()
        elif self.task_type == 'multiple_choice':
            if self.metric == 'Accuracy':
                return self.eval_accuracy()
            elif self.metric == 'F1':
                return self.eval_f1()
            

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

# Define objective tasks based on lexeval format (from evaluate.py logic)
# As per instructions: multiple_choice -> traditional, generation -> LLM
# I need to infer which task is multiple_choice. The evaluate.py shows:
# 1_2, 2_1, 2_2, 2_4, 3_1, 3_2, 3_3, 4_1, 4_2 are multiple_choice (usually these output "Accuracy")
# Let's map it based on common lexeval setups, or we can look inside the files if needed.
# From evaluate.py, it seems we might need to know the task type.
# A safe way is to check the format of the output/answer, but let's define the standard LexEval objective tasks:
OBJECTIVE_TASKS = {
    "1_2", "2_1", "2_2", "2_4", "3_1", "3_2", "3_3", "4_1", "4_2"
}

# =====================================================================
# [Configuration] Batch Processing
# Paths to the old LLM trajectory directories
BATCH_TRAJ_DIRS = [
    "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/LexEval/evaluation_output/deepseek-v4-flash_scored",
]

# Default output directory for the fixed result files
DEFAULT_OUTPUT_DIR = "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/bench_result/lexeval"
# =====================================================================

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
        print(f"[WARN] 目录 {old_traj_dir} 中未扫描到任何 *.jsonl 轨迹文件，跳过该项。")
        return

    for file_path in jsonl_files:
        task_name = os.path.basename(file_path).replace(".jsonl", "")
        items_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    items_list.append(json.loads(line))
        
        count = len(items_list)
        if count == 0: continue
        
        # === LLM Pipeline Scores ===
        t_llm_score = sum(item.get("eval_score", 0) for item in items_list)
        t_time = sum(item.get("metrics", {}).get("total_time_sec", 0) for item in items_list)
        t_tool = sum(item.get("metrics", {}).get("tool_latency_sec", 0) for item in items_list)
        t_rag = sum(item.get("metrics", {}).get("rag_count", 0) for item in items_list)
        t_tok = sum(item.get("metrics", {}).get("total_tokens", 0) for item in items_list)
        t_user = sum(item.get("metrics", {}).get("user_prompt_tokens", 0) for item in items_list)
        t_inter = sum(item.get("metrics", {}).get("inter_agent_tokens", 0) for item in items_list)
        t_comp = sum(item.get("metrics", {}).get("completion_tokens", 0) for item in items_list)

        # === Traditional Scoring using evaluate.py ===
        trad_score = None
        trad_abstention = None
        is_objective = task_name in OBJECTIVE_TASKS

        if Evaluator is not None:
            try:
                if is_objective:
                    evaluator = Evaluator(file_path=file_path, task_type='multiple_choice', metric='Accuracy')
                    trad_score_val = evaluator.eval()
                    trad_score = trad_score_val * 100.0 # Convert to percentage
                else:
                    evaluator = Evaluator(file_path=file_path, task_type='generation', metric='Rouge_L')
                    trad_score_val = evaluator.eval()
                    trad_score = trad_score_val * 100.0
            except Exception as e:
                print(f"\n[ERROR] 任务 {task_name} 的传统评分计算失败:")
                traceback.print_exc()

        # === Hybrid Score Calculation ===
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
            if "_scored" in folder_name:
                base_model_name = folder_name.split("_scored")[0]
            else:
                base_model_name = folder_name
                
            target_filename = f"{base_model_name}_lexeval_fix.json"
            computed_output_path = os.path.join(DEFAULT_OUTPUT_DIR, target_filename)
            
            print(f"\n⚡ 正在处理: {folder_name} -> 目标文件名: {target_filename}")
            run_single_evaluation(traj_dir, computed_output_path)
        print("\n✨ 所有批量任务均已执行完毕。")
    else:
        print("ℹ️ [硬编码列表为空] 正在自动回归至旧有的命令行单条处理模式...")
        if not args.old_traj_dir or not args.new_result_path:
            print("\n[ERROR] 回归旧单任务模式失败：未通过命令行检测到 --old_traj_dir 或 --new_result_path 参数！")
            sys.exit(1)
        run_single_evaluation(args.old_traj_dir, args.new_result_path)

if __name__ == "__main__":
    main()