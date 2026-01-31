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
如果AI助手 1 表现更好，输出“[[1]]”；如果AI助手 2 表现更好，输出“[[2]]”；如果平局，输出“[[3]]”。 /no_think"""

class VLLMRewardManager:
    def __init__(self, api_url="http://localhost:9000/v1", api_key="EMPTY"):
        # 使用异步客户端
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_url)
        self.model_name = "qwen3-8b-reward" # 需与 vLLM 启动名一致

    def parse_score(self, completion_text: str, swapped: bool) -> float:
        match = re.findall(r"\[\[(\d)\]\]", completion_text)
        if not match: return 0.0
        
        label = match[-1]
        
        # 核心逻辑：
        # 如果没有交换 (d1=GT, d2=Model)：label "2" 是模型赢 (+1), label "1" 是模型输 (-1)
        # 如果交换了 (d1=Model, d2=GT)：label "1" 是模型赢 (+1), label "2" 是模型输 (-1)
        
        if label == "3": # 平局
            return 0.0
            
        if not swapped:
            return 1.0 if label == "2" else -1.0
        else:
            return 1.0 if label == "1" else -1.0

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
                max_tokens=48,
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
    # 获取或创建事件循环
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # 关键：如果在异步环境运行（如测试脚本里有 asyncio.run）
    # 使用以下方式可以避免嵌套 loop 错误
    if loop.is_running():
        # 这里需要用到 nest_asyncio 或者改写调用逻辑
        # 对于 verl，它通常从同步 worker 调用，所以下面这行通常足够：
        import nest_asyncio
        nest_asyncio.apply()
    
    return loop.run_until_complete(manager.batch_get_rewards(data_batch))

def compute_score(solution_str, ground_truth, extra_info=None):
    """
    verl __init__.py 调用此接口进行单条判分 。
    solution_str: 模型生成的回答。
    ground_truth: Parquet 中的 golden_answers 。
    extra_info: Parquet 中的 extra_info 字典 。
    """
    # 字段映射：从 Parquet 结构提取数据 
    # 1. 提取需求：优先从 extra_info 字典拿 question 字段
    needs = extra_info.get('question', "请根据要求判分") if extra_info else "请根据要求判分"
    
    # 2. 提取参考答案：Parquet 中 golden_answers 是列表，取第一个元素 
    reference = ground_truth[0] if isinstance(ground_truth, list) and len(ground_truth) > 0 else str(ground_truth)

    # 异步环境适配 
    import nest_asyncio
    nest_asyncio.apply()
    
    manager = VLLMRewardManager()
    loop = asyncio.get_event_loop()
    
    # 调用现有的异步判分逻辑
    score = loop.run_until_complete(manager.get_reward_async(
        needs=needs,
        ground_truth=reference,
        model_output=solution_str
    ))
    
    return score