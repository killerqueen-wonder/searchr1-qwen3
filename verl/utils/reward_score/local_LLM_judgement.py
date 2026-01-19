import asyncio
import re
import random
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
如果AI助手 1 表现更好，输出“[[1]]”；如果AI助手 2 表现更好，输出“[[2]]”；如果平局，输出“[[3]]”。"""

class VLLMRewardManager:
    def __init__(self, api_url="http://localhost:9000/v1", api_key="EMPTY"):
        # 使用异步客户端
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_url)
        self.model_name = "qwen3-8b-reward" # 需与 vLLM 启动名一致

    def parse_score(self, completion_text: str, swapped: bool) -> float:
        """
        解析模型输出并计算 Reward。
        RL 需要标量分，我们将判定结果映射到 [0, 1] 或 [-1, 1]。
        """
        try:
            # 匹配最后一个 [[x]]
            match = re.findall(r"\[\[(\d)\]\]", completion_text)
            if not match:
                return 0.0  # 解析失败给中性分
            
            label = match[-1]
            
            # 定义基础分：1是更好，2是更差，3是平局
            # 注意：这里的 1 和 2 是相对于 Prompt 里的顺序的
            score_map = {
                "1": 1.0,  # 助手1胜
                "2": -1.0, # 助手2胜
                "3": 0.0   # 平局
            }
            raw_score = score_map.get(label, 0.0)

            # 如果我们在构造请求时交换了顺序，则分号也需要取反
            # 目的是让 Reward 始终代表“当前模型生成结果”的质量
            return -raw_score if swapped else raw_score

        except Exception as e:
            print(f"Parsing error: {e}")
            return 0.0

    async def get_reward_async(self, needs: str, ground_truth: str, model_output: str) -> float:
        """单条数据的异步判分逻辑"""
        # 随机交换顺序以消除位置偏见 (Position Bias)
        swapped = random.random() < 0.5
        d1, d2 = (model_output, ground_truth) if swapped else (ground_truth, model_output)
        
        prompt = EVALUATION_PROMPT.format(needs=needs, dialogue1=d1, dialogue2=d2)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, # 判分必须用 0 温度保证稳定性
                max_tokens=512,
                stop=["]]"] # 提前截断，节省 Token
            )
            # 由于加了 stop，输出可能是 "...[[1"
            full_text = response.choices[0].message.content + "]]"
            return self.parse_score(full_text, swapped)
        except Exception as e:
            print(f"API Call Error: {e}")
            return 0.0

    async def batch_get_rewards(self, batch_data: List[Dict]) -> List[float]:
        """批量异步处理一个 RL Step 的所有数据"""
        tasks = []
        for item in batch_data:
            # verl 传入的数据格式通常包含 prompt (user needs) 和 response (model output)
            # 以及你在 dataset 中准备的 ground_truth
            tasks.append(self.get_reward_async(
                needs=item['prompt'],
                ground_truth=item['reference'],
                model_output=item['responses']
            ))
        
        # 并发执行
        return await asyncio.gather(*tasks)

# 对接 verl 的入口函数
def reward_score_fn(data_batch):
    """
    verl 的 RewardManager 会调用此函数
    data_batch: verl 提供的包含输入输出的 batch
    """
    manager = VLLMRewardManager()
    loop = asyncio.get_event_loop()
    
    # 执行异步任务并返回结果列表
    scores = loop.run_until_complete(manager.batch_get_rewards(data_batch))
    return scores