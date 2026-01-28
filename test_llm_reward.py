import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import json


import sys
import os

# 1. 指向包含该文件的“目录”路径（注意：不要带文件名）
target_dir = "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/searchr1-qwen3/verl/utils/reward_score"

# 2. 加入系统路径
if target_dir not in sys.path:
    sys.path.append(target_dir)

# 3. 直接引用文件名（不带 .py）

from local_LLM_judgement import VLLMRewardManager, reward_score_fn

# 模拟 OpenAI 的响应结构
class MockResponse:
    def __init__(self, content):
        # 模拟 response.choices[0].message.content
        self.choices = [
            MagicMock(message=MagicMock(content=content))
        ]

async def test_reward_manager():
    print("开始测试 Reward Manager 逻辑...")
    
    test_batch = [
        {"prompt": "A", "reference": "RefA", "responses": "RespA"},
        {"prompt": "B", "reference": "RefB", "responses": "RespB"}
    ]

    # ✅ 正确做法：构造模拟的返回数据对象（不是 AsyncMock）
    res1 = MockResponse("经过分析，助手 1 更好。[[1]]")
    res2 = MockResponse("两个助手表现差不多。[[3]]")

    # 注意路径：根据你之前的路径设置
    with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        # 让异步方法返回我们构造好的对象
        mock_create.side_effect = [res1, res2]
        
        from local_LLM_judgement import reward_score_fn
        scores = reward_score_fn(test_batch)
        
        print(f"测试完成！获取到的分数为: {scores}")
        
        # 验证逻辑
        if scores == [0.0, 0.0]:
            print("❌ 测试未命中逻辑（得分全为0），请检查 API 调用是否报错。")
        else:
            print("✅ 逻辑验证成功！")

if __name__ == "__main__":
    # 确保使用同步方式调用已经在内部处理了 loop 的函数
    test_reward_manager()