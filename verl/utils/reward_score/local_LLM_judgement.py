import asyncio
import re
import random
import concurrent.futures
from typing import List, Dict
from openai import AsyncOpenAI

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

class VLLMRewardManager:
    def __init__(self, api_url="http://localhost:9000/v1", api_key="EMPTY"):
        # 使用异步客户端
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_url)
        self.model_name = "qwen3-8b-reward" # 需与 vLLM 启动名一致

    def parse_score(self, completion_text: str, swapped: bool) -> float:
        """解析模型输出并映射为分值 """
        match = re.findall(r"\[\[(\d)\]\]", completion_text)
        if not match: return 0.0
        
        label = match[-1]
        
        if label == "3": # 平局
            return 0.0
            
        if not swapped:
            # 助手1是GT，助手2是模型。label "2" 代表模型赢 
            return 1.0 if label == "2" else -1.0
        else:
            # 助手1是模型，助手2是GT。label "1" 代表模型赢 
            return 1.0 if label == "1" else -1.0

    async def get_reward_async(self, needs: str, ground_truth: str, model_output: str) -> float:
        """单条数据的异步判分逻辑 """
        swapped = random.random() < 0.5
        d1, d2 = (model_output, ground_truth) if swapped else (ground_truth, model_output)
        
        prompt = EVALUATION_PROMPT.format(needs=needs, dialogue1=d1, dialogue2=d2)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=48, # 限制输出长度以节省显存 
                stop=["]]"]
            )
            full_text = response.choices[0].message.content + "]]"
            return self.parse_score(full_text, swapped)
        except Exception as e:
            print(f"API Call Error: {e}")
            return 0.0

def compute_score(solution_str, ground_truth, extra_info=None):
    """
    verl 同步 Worker 调用此接口。
    通过在独立线程中运行 asyncio.run 来绕过 uvloop 冲突 。
    """
    # 字段映射逻辑
    needs = extra_info.get('question', "请根据要求判分") if extra_info else "请根据要求判分"
    reference = ground_truth[0] if isinstance(ground_truth, list) and len(ground_truth) > 0 else str(ground_truth)
    
    manager = VLLMRewardManager()
    coro = manager.get_reward_async(
        needs=needs,
        ground_truth=reference,
        model_output=solution_str
    )

    try:
        # 尝试直接运行（如果当前线程没有运行中的 loop）
        return asyncio.run(coro)
    except RuntimeError:
        # 核心修复：如果当前是在 uvloop 中运行的同步代码，开启新线程执行异步任务 
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()