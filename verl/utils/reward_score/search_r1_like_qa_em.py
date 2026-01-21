# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py
import random
import re
import string
import unicodedata

def normalize_answer(s):
    if s is None: # 防御性编程：处理 None 输入
        return ""
    
    def white_space_fix(text):
        return " ".join(text.split())

    
    def remove_punc_keep_numeric(text):
        numeric_whitelist = {'.', '-', '%'}
        res = []
        for char in text:
            cat = unicodedata.category(char)
            if char.isalnum() or char.isspace():
                res.append(char)
            elif cat.startswith('P'):
                if char in numeric_whitelist:
                    res.append(char)
                else:
                    res.append(" ") # 将其他标点替换为空格，避免粘连
        return "".join(res)

    def lower(text):
        return text.lower()

    # 移除了会导致 ABCD 错误的 remove_articles
    return white_space_fix(remove_punc_keep_numeric(lower(s)))

def em_check(prediction, golden_answers, max_score=1.0):
    if prediction is None:
        return 0
    
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    
    # 归一化标准答案
    gold_set = {normalize_answer(g) for g in golden_answers if normalize_answer(g)}

    # 归一化预测答案
    norm_pred = normalize_answer(prediction)
    
    # 策略：如果标准答案里所有项都不含空格，则按空格 split 进行集合比较（适合 A B C D）
    # 否则，尝试逗号切分或整体匹配（适合短语）
    if all(" " not in g for g in gold_set):
        pred_set = set(norm_pred.split())
    else:
        # 如果包含短语，建议模型输出用逗号分隔，这里做简单兼容处理
        pred_set = {p.strip() for p in re.split(r'[,，]', norm_pred) if p.strip()}

    if not pred_set:
        return 0

    if pred_set == gold_set:
        return max_score
    
    if pred_set.issubset(gold_set):
        return max_score / 5
    
    return 0

def extract_solution(solution_str):
    if not solution_str:
        return None
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()

# def wrong_format(text):
#     return any([
#         text.count("<answer>") > 4, 
#         text.count("</answer>") > 4,
#         text.count("<search>") != text.count("</search>"),
#         text.count("<search>") != text.count("<information>")
#     ])

def correct_format(text):
    
    return all([
        text.count("<answer>") <= 4, 
        text.count("</answer>") <= 4,
        text.count("<search>") == text.count("</search>"),
        text.count("<search>") == text.count("<information>"),
        text.count("<search>") >=1
        
    ])

def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    answer = extract_solution(solution_str=solution_str)
    
    
    # 只有当开启随机采样时才打印，避免日志溢出
    if random.randint(1, 64) == 1:
        print("---------------start-----------------")
        print(f"Golden answers: {ground_truth.get('target')}")
        print(f"Extracted answer: {answer}")
        print("---------------end-----------------")

    # 1. 如果根本没提取到答案
    if answer is None:
        return -0.1

    if len(solution_str) < 10:#思考过程太短
        return 0  
    
    # 2. 计算内容得分 
    final_score = em_check(answer, ground_truth.get("target", []), score)
    
    if final_score > 0:
        # 答案正确（或部分正确），但格式错误 -> 降级处罚
        if not correct_format(solution_str):
            return final_score / 4
        return final_score
    else:
        # 答案错误，但格式完全正确 -> 给予微小鼓励分
        if correct_format(solution_str):
            return format_score
        return 0
