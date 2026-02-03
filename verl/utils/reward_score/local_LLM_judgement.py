import asyncio
import re
import random
import concurrent.futures
from typing import List, Dict
from openai import OpenAI

# 保持与你原始 Prompt 一致的评估模板
EVALUATION_PROMPT = """扮演一个公正的评委，评估用户与两个AI助手之间的对话，以判断哪个AI助手为用户提供的服务更好。
用户的需求是：
{needs}

[AI助手1对话开始]
{dialogue1}
[AI助手1对话结束]

[AI助手2对话开始]
{dialogue2}
[AI助手2对话结束]

你的评估需要考虑AI助手回复是否满足以下标准：1.准确，专业。2.匹配问题，完整作答。3.逻辑清晰，推理合理。4.实用。
请先针对以上规定细致分析，最后严格按照以下格式输出结论：
如果AI助手 1 表现更好，输出“[[1]]”；如果AI助手 2 表现更好，输出“[[2]]”；如果平局，输出“[[3]]”。 /no_think"""

# --- 单例模式管理客户端 ---
_GLOBAL_CLIENT = None

def get_client(api_url="http://localhost:9000/v1"):
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        _GLOBAL_CLIENT = OpenAI(api_key="EMPTY", base_url=api_url)
    return _GLOBAL_CLIENT

class VLLMRewardManager:
    def __init__(self, api_url="http://localhost:9000/v1"):
        self.client = get_client(api_url)
        self.model_name = "qwen3-8b-reward"

    def parse_score(self, completion_text: str, swapped: bool) -> float:
        match = re.findall(r"\[\[(\d)\]\]", completion_text)
        if not match: return 0.0
        label = match[-1]
        if label == "3": return 0.5
        if not swapped:
            return 1.0 if label == "2" else -1.0
        else:
            return 1.0 if label == "1" else -1.0

    def get_reward_sync(self, needs: str, ground_truth: str, model_output: str) -> float:
        """同步判分逻辑，彻底规避协程冲突"""
        swapped = random.random() < 0.5
        d1, d2 = (model_output, ground_truth) if swapped else (ground_truth, model_output)
        prompt = EVALUATION_PROMPT.format(needs=needs, dialogue1=d1, dialogue2=d2)

        try:
            # 直接使用同步调用
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=48,
                stop=["]]"]
            )
            full_text = response.choices[0].message.content + "]]"
            return self.parse_score(full_text, swapped)
        except Exception as e:
            print(f"API Call Error: {e}")
            return 0.0

def extract_solution(solution_str):#抽取最终答案
    if not solution_str:
        return None
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()

def correct_format(text):
    
    return all([
        text.count("<answer>") <= 4, 
        text.count("</answer>") <= 4,
        text.count("<search>") == text.count("</search>"),
        text.count("<search>") == text.count("<information>"),
        text.count("<search>") >=1
        
    ])

# --- verl 调用的核心接口 ---
def compute_score(solution_str, ground_truth, extra_info=None):
    """
    现在是纯同步实现，不再需要 nest_asyncio 或 ThreadPoolExecutor。
    """
    needs = extra_info.get('question', "请根据要求判分") if extra_info else "请根据要求判分"
    # needs=''
    reference = ground_truth[0] if isinstance(ground_truth, list) and len(ground_truth) > 0 else str(ground_truth)
    
    #只取最终答案部分
    answer = extract_solution(solution_str=solution_str)

    # 1. 如果根本没提取到答案
    if answer is None:
        return -1

    if len(solution_str) < 20:#思考过程太短
        return 0  
    # 实例化 Manager（实际上复用了全局 Client）
    manager = VLLMRewardManager()

    final_score=manager.get_reward_sync(
        needs=needs,
        ground_truth=reference,
        model_output=answer
    )

    if final_score == 1:
        # 答案正确（或部分正确），但格式错误 -> 降级处罚
        if not correct_format(solution_str):
            final_score -= 0.2
        
    else:
        # 答案错误，但格式完全正确 -> 给予微小鼓励分
        if correct_format(solution_str):
            final_score += 0.2
        
    
    # 只有当开启随机采样时才打印，避免日志溢出
    if random.randint(1, 64) == 1:
        print("---------------start-----------------")
        print(f"Golden answers: {reference}")
        print(f"Extracted answer: {answer}")
        print(f"final_score: {final_score}")
        print("---------------end-----------------")


    
    # 直接同步获取分数
    return final_score