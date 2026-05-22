import json
import os
import argparse
import glob
import sys
import traceback
import tempfile
import re

import jsonlines
import torch
import torch.nn as nn
import pandas as pd
import bert_score
import jieba
from transformers import BertTokenizer, BartForConditionalGeneration
from typing import List
from rouge import Rouge
from process import BARTScorer, find_valid_substrings, normalize_zh_answer
from tqdm import tqdm

import numpy as np
sys.setrecursionlimit(20000)

# =====================================================================
# 内联 Evaluator 类：完美规避第三方库包名冲突
# =====================================================================
class Evaluator:
    '''
    Evaluating the score for a given task on a given metric from one model's output
    '''
    def __init__(self, file_path, task_type, metric, device='cpu', model_path=None):
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
                    raise ValueError(f"Lacking bert model path for bertscore")
            elif metric == 'Bartscore':
                self.metric = metric
                if model_path != None:
                    self.model_path = model_path
                else:
                    raise ValueError(f"Lacking bart model path for bartscore")
            else:
                raise ValueError(f"No metric {metric} for generation task!")

        elif task_type == 'multiple_choice':
            self.task_type = task_type
            if metric == 'Accuracy':
                self.metric = metric
            elif metric == 'F1':
                self.metric = metric
            else:
                raise ValueError(f"No metric {metric} for multiple choice task!")
        else:
            raise ValueError(f"No task type {task_type}")

        self.device = device
        
    def read_jsonl(self):
        with jsonlines.open(self.file_path, "r") as r:
            for obj in r:
                yield obj
                
    def eval_accuracy(self):
        total_number = 0
        correct_number = 0
        for qa_one in self.read_jsonl():
            if qa_one['output'] == "":
                ans = ""
            else:
                ans = find_valid_substrings(qa_one['output'])
            if qa_one['answer'] == ans:
                correct_number += 1
            total_number += 1
        if total_number == 0:
            return 0
        return correct_number/total_number

    def eval_f1(self):
        total_number = 0
        score = 0
        for qa_one in self.read_jsonl():
            if qa_one['output'] == "":
                ans = ""
            else:
                ans = find_valid_substrings(qa_one['output'])
            s1 = set(ans)
            s2 = set(qa_one['answer'])
            if len(s1) == 0:
                p = 0
            else:
                p = len(s1 & s2) / len(s1)
            
            if len(s2) == 0:
                r = 0
            else:
                r = len(s1 & s2) / len(s2)
            
            if p+r == 0:
                f1 = 0
            else:
                f1 = 2*p*r/(p+r)
            score += f1
            total_number += 1
        if total_number == 0:
            return 0
        return score/total_number

    def eval_rougel(self):
        total_number = 0
        score = 0
        rouge = Rouge()
        for qa_one in self.read_jsonl():
            if qa_one['output'] == "":
                ans = "没有 答案"
            else:
                ans = " ".join(list(jieba.cut(normalize_zh_answer(qa_one['output']), cut_all=False)))
            
            ref = " ".join(list(jieba.cut(normalize_zh_answer(qa_one['answer']), cut_all=False)))
            
            if len(ans) == 0:
                ans = "没有 答案"
            if len(ref) == 0:
                ref = "没有 答案"
            s = rouge.get_scores(ans, ref)
            score += s[0]['rouge-l']['f']
            total_number += 1
        if total_number == 0:
            return 0
        return score/total_number

    def eval_bertscore(self):
        import logging
        import transformers
        transformers.tokenization_utils.logger.setLevel(logging.ERROR)
        transformers.configuration_utils.logger.setLevel(logging.ERROR)
        transformers.modeling_utils.logger.setLevel(logging.ERROR)

        preds = []
        refs = []
        for qa_one in self.read_jsonl():
            if qa_one['output'] == "":
                preds.append("没有答案")
            else:
                preds.append(qa_one['output'])
            refs.append(qa_one['answer'])
        _, _, F1 = bert_score.score(preds, refs, model_type=self.model_path, num_layers=9, device=self.device)
        return F1.mean().item()

    def eval_bartscore(self):
        bart_scorer = BARTScorer(device=self.device, checkpoint=self.model_path)
        preds = []
        refs = []
        for qa_one in self.read_jsonl():
            if qa_one['output'] == "":
                preds.append("没有答案")
            else:
                preds.append(qa_one['output'])
            refs.append(qa_one['answer'])
        score = bart_scorer.score(preds, refs)
        return np.mean(score)
            
    def eval(self):
        if self.metric == 'Accuracy':
            return self.eval_accuracy()
        elif self.metric == 'F1':
            return self.eval_f1()
        elif self.metric == 'Rouge_L':
            return self.eval_rougel()
        elif self.metric == 'Bertscore':
            return self.eval_bertscore()
        elif self.metric == 'Bartscore':
            return self.eval_bartscore()
            
# =====================================================================
# 常量及参数配置
# =====================================================================

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
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/LexEval/evaluation_output/A2Search-0516_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/LexEval/evaluation_output/lawgpt_0517_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/LexEval/evaluation_output/legalone-0517_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/LexEval/evaluation_output/luwen-0517_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/LexEval/evaluation_output/qwen3-8b-0503_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/LexEval/evaluation_output/qwen3-8b-AR-0520_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/LexEval/evaluation_output/qwen3-8b-AR-0521_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/LexEval/evaluation_output/qwen3-8b-no-RAG-0519-0143_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/LexEval/evaluation_output/qwen3-8b-SFT-AR-0521_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/LexEval/evaluation_output/qwen3-post-train-0519_scored',
    '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/LexEval/evaluation_output/R-Search-0516_scored'
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

        if Evaluator is not None:
            try:
                # 恢复了全量传统评分：无论是多选还是生成题，都会算出自己的 trad_score
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

        # 核心：半传统融合打分规则，客观题采用传统分数，主观题采用大模型分数
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
            
            # ===============================================================
            # 新增逻辑：提前检查目标输出路径是否存在
            # 如果存在则打印跳过信息，进入下一个循环，节约时间
            # ===============================================================
            if os.path.exists(computed_output_path):
                print(f"\n⏭️  [缓存命中] 已检测到目标文件存在: {target_filename}，避免重复计算，正在跳过...")
                continue
            
            print(f"\n⚡ 正在处理: {folder_name} -> 目标文件名: {target_filename}")
            run_single_evaluation(traj_dir, computed_output_path)
        print("\n✨ 所有批量任务均已执行完毕。")
    else:
        run_single_evaluation(args.old_traj_dir, args.new_result_path)

if __name__ == "__main__":
    main()