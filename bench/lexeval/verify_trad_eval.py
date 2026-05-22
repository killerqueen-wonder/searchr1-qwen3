import json
import os
import argparse
import glob
import sys
import re
from collections import OrderedDict
import traceback

# 1. 放开递归限制，防止 Rouge-L 计算长文本时崩溃
sys.setrecursionlimit(20000)

import jieba
from rouge import Rouge

# =====================================================================
# 底层文本处理与提取逻辑 (完全复刻 evaluate.py & process.py)
# =====================================================================
def find_valid_substrings(s):
    if not s:
        return ""
    # 截断解析部分
    s = s.split('解析')[0].split('分析')[0]
    # 清洗标点
    s = s.replace("、", "").replace(".", "").replace(",", "").replace(";", "").replace("，", "").replace("和", "").replace(", ", "")
    # 正则提取 ABCDE 组合
    pattern = r'[ABCDE]{1,5}'
    substrings = re.findall(pattern, s)
    # 过滤重复字符块，合并并去重
    valid_substrings = [sub for sub in substrings if len(sub) == len(set(sub))]
    valid_substrings = "".join(valid_substrings)
    valid_substrings = ''.join(OrderedDict.fromkeys(valid_substrings))
    return valid_substrings

def normalize_zh_answer(s):
    if s is None:
        return ""
    def white_space_fix(text):
        return "".join(text.split())
    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        en_punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        punctuation = cn_punctuation + en_punctuation
        return "".join(ch for ch in text if ch not in punctuation)
    def lower(text):
        if text is None:
            return ""
        return str(text).lower()
    return white_space_fix(remove_punc(lower(s)))


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
    parser = argparse.ArgumentParser(description="LexEval 传统评分白盒验证脚本")
    parser.add_argument('--input_dir', type=str, required=True, help="含有 LLM 评分的 _scored 目录")
    parser.add_argument('--output_dir', type=str, required=True, help="验证结果输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 动态推导模型名称（自动剔除 _scored 后缀，保持命名整洁）
    raw_model_name = os.path.basename(os.path.normpath(args.input_dir))
    model_name = raw_model_name.replace("_scored", "")
    
    jsonl_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"[WARN] 输入目录 {args.input_dir} 中未扫描到任何 *.jsonl 文件。")
        return

    rouge_scorer = Rouge()
    
    global_stats = {
        "total_count": 0, "total_llm_score": 0.0, "total_hybrid_score": 0.0, 
        "total_trad_score": 0.0, "trad_sample_size": 0
    }
    category_totals = {
        cat: {
            "total_llm_score": 0.0, "total_hybrid_score": 0.0, "sample_size": 0, 
            "total_trad_score": 0.0, "trad_sample_size": 0
        } for cat in set(CATEGORY_MAPPING.values())
    }
    task_results = {}

    print(f"🚀 开始白盒验证传统评分... 模型: {model_name}")

    for file_path in jsonl_files:
        raw_name = os.path.basename(file_path).replace(".jsonl", "")
        # 正则提取真实任务ID
        match = re.search(r'(\d+_\d+)$', raw_name)
        task_name = match.group(1) if match else raw_name
        
        is_subjective = task_name.startswith("5_")
        is_objective = not is_subjective
        task_category = CATEGORY_MAPPING.get(task_name, "unknown")
        
        processed_items = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                
                # 1. 踢除 metrics 字段，让文件变清爽
                item.pop("metrics", None)
                
                ans = item.get("output", "")
                ref = item.get("answer", "")
                if ref is None: ref = ""
                if ans is None: ans = ""
                
                # 2. 传统评分与字面抽取
                if is_objective:
                    extracted = find_valid_substrings(ans)
                    trad_score = 100.0 if extracted == ref else 0.0
                    item["extracted_ans"] = extracted
                    item["trad_score"] = trad_score
                else:
                    norm_ans = " ".join(list(jieba.cut(normalize_zh_answer(ans), cut_all=False))) if ans else "没有 答案"
                    norm_ref = " ".join(list(jieba.cut(normalize_zh_answer(ref), cut_all=False))) if ref else "没有 答案"
                    if len(norm_ans.strip()) == 0: norm_ans = "没有 答案"
                    if len(norm_ref.strip()) == 0: norm_ref = "没有 答案"
                    try:
                        scores = rouge_scorer.get_scores(norm_ans, norm_ref)
                        trad_score = scores[0]['rouge-l']['f'] * 100.0
                    except:
                        trad_score = 0.0
                    item["extracted_ans"] = norm_ans 
                    item["trad_score"] = trad_score
                
                processed_items.append(item)

        count = len(processed_items)
        if count == 0: continue
        
        # 落盘新格式的 jsonl 文件
        output_jsonl_path = os.path.join(args.output_dir, os.path.basename(file_path))
        with open(output_jsonl_path, 'w', encoding='utf-8') as out_f:
            for item in processed_items:
                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        # 3. 统计汇总 (完美读取已有的 LLM Score)
        t_llm_score = sum(item.get("eval_score", 0) for item in processed_items)
        t_trad_score = sum(item["trad_score"] for item in processed_items)
        
        # 融合分数逻辑：客观题用传统分，主观题用LLM分
        t_hybrid_score = t_trad_score if is_objective else t_llm_score

        task_results[task_name] = {
            "category": task_category,
            "task_type": "subjective" if is_subjective else "objective",
            "avg_score": t_hybrid_score / count,
            "llm_judge_avg_score": t_llm_score / count,
            "traditional_avg_score": t_trad_score / count,
            "traditional_abstention_rate": 0,
            "avg_time": 0.0, "avg_tool_latency": 0.0, "avg_rag": 0.0,
            "avg_total_tokens": 0.0, "avg_user_tokens": 0.0, "avg_inter_tokens": 0.0, "avg_comp_tokens": 0.0,
            "sample_size": count
        }

        global_stats["total_count"] += count
        global_stats["total_llm_score"] += t_llm_score
        global_stats["total_hybrid_score"] += t_hybrid_score
        global_stats["total_trad_score"] += t_trad_score
        global_stats["trad_sample_size"] += count

        if task_category in category_totals:
            cat_stats = category_totals[task_category]
            cat_stats["total_llm_score"] += t_llm_score
            cat_stats["total_hybrid_score"] += t_hybrid_score
            cat_stats["sample_size"] += count
            cat_stats["total_trad_score"] += t_trad_score
            cat_stats["trad_sample_size"] += count
            
    # 4. 生成大类与最终 Result 报告
    category_breakdown = {}
    for cat, stats in category_totals.items():
        sz = stats["sample_size"]
        if sz == 0: continue
        category_breakdown[cat] = {
            "avg_score": stats["total_hybrid_score"] / sz,           
            "llm_judge_avg_score": stats["total_llm_score"] / sz,    
            "traditional_avg_score": stats["total_trad_score"] / sz,
            "traditional_abstention_rate": 0,
            "avg_time": 0.0, "avg_tool_latency": 0.0, "avg_rag": 0.0,
            "avg_total_tokens": 0.0, "avg_user_tokens": 0.0, "avg_inter_tokens": 0.0, "avg_comp_tokens": 0.0,
            "sample_size": sz
        }

    g_count = global_stats["total_count"] if global_stats["total_count"] > 0 else 1
    
    final_report = {
        "bench_name": "LexEval_Trad_Verify",
        "global_average_score": global_stats["total_hybrid_score"] / g_count,
        "global_llm_judge_score": global_stats["total_llm_score"] / g_count,
        "traditional_global_average_score": global_stats["total_trad_score"] / g_count,
        "traditional_global_abstention_rate": 0,
        "efficiency": {
            "avg_end_to_end_time": 0.0, "avg_tool_latency": 0.0, "avg_rag_rounds": 0.0,
            "avg_total_tokens": 0.0, "avg_user_tokens": 0.0, "avg_inter_tokens": 0.0, "avg_comp_tokens": 0.0
        },
        "category_breakdown": category_breakdown,
        "task_breakdown": task_results
    }

    result_json_path = os.path.join(args.output_dir, f"{model_name}_lexeval_result.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ 传统评分白盒验证完毕！")
    print(f"📁 包含 eval_score 与 trad_score 对比的轨迹文件已保存至: {args.output_dir}")
    print(f"📊 最终聚合报告已保存至: {result_json_path}")

if __name__ == "__main__":
    main()