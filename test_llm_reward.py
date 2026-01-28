import asyncio
from unittest.mock import AsyncMock, patch
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

async def test_reward_manager():
    print("开始测试 Reward Manager 逻辑...")
    
    # 1. 构造模拟数据 (符合 verl 格式)
    test_batch = [
        {
            "prompt": "请解释什么是量子纠缠。",
            "reference": "量子纠缠是量子力学中的一种现象，两个粒子即使相距遥远也会保持关联。",
            "responses": "量子纠缠就是两个微观粒子之间存在一种‘心灵感应’，一个变了另一个立刻也变。"
        },
        {
            "prompt": "如何写一个 Python 的 hello world？",
            "reference": "使用 print('Hello, World!')",
            "responses": "你可以输入 print('Hello, World!') 来实现。"
        }
    ]

    # 2. 模拟 vLLM 的 API 返回
    # 模拟第一次：模型 1 胜 [[1]]
    mock_res1 = AsyncMock()
    mock_res1.choices = [AsyncMock(message=AsyncMock(content="经过分析，助手 1 更好。[[1]]"))]
    
    # 模拟第二次：平局 [[3]]
    mock_res2 = AsyncMock()
    mock_res2.choices = [AsyncMock(message=AsyncMock(content="两个助手表现差不多。[[3]]"))]

    # 3. 使用 patch 拦截 API 调用
    with patch("openai.resources.chat.completions.AsyncCompletions.create") as mock_create:
        mock_create.side_effect = [mock_res1, mock_res2]
        
        # 执行测试
        scores = reward_score_fn(test_batch)
        
        print(f"测试完成！获取到的分数为: {scores}")
        
        # 验证结果
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        print("✅ 基础流程测试通过！")

if __name__ == "__main__":
    asyncio.run(test_reward_manager())