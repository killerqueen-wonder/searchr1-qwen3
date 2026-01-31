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
        if label == "3": return 0.0
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

# --- verl 调用的核心接口 ---
def compute_score(solution_str, ground_truth, extra_info=None):
    """
    现在是纯同步实现，不再需要 nest_asyncio 或 ThreadPoolExecutor。
    """
    needs = extra_info.get('question', "请根据要求判分") if extra_info else "请根据要求判分"
    reference = ground_truth[0] if isinstance(ground_truth, list) and len(ground_truth) > 0 else str(ground_truth)
    
    # 实例化 Manager（实际上复用了全局 Client）
    manager = VLLMRewardManager()
    
    # 直接同步获取分数
    return manager.get_reward_sync(
        needs=needs,
        ground_truth=reference,
        model_output=solution_str
    )