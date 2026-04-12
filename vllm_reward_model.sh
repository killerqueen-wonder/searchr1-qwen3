#!/bin/bash

# 1. 设置使用的显卡 (使用第4张 A800)
export CUDA_VISIBLE_DEVICES=1

# 2. 模型路径
# MODEL_PATH="/F00120250029/lixiang_share/Models/Qwen3-4B-Instruct-2507" # 请修改为你的实际路径
MODEL_PATH="/data/panghuaiwen/legal_R1/model/Qwen/Qwen3-8B" # 请修改为你的实际路径
# MODEL_PATH="/F00120250029/lixiang_share/Models/Qwen3-8B" # 请修改为你的实际路径

# 3. 启动 vLLM API 服务
# 建议参数说明：
# --host 0.0.0.0: 允许外部访问
# --port 8000: 与 Python 脚本中的 api_url 对应
# --served-model-name: 必须与脚本中 self.model_name = "qwen3-8b-reward" 保持一致
# --enable-prefix-caching: 核心优化！RL 判分时 System Prompt 相同，开启此项速度提升巨大
# --max-num-seqs: 对应 RL 训练的 Rollout Batch Size，建议设为 128 或更大
# --max-model-len: 限制模型窗口，判分不需要太长，4096 足够，能节省大量显存给 KV Cache
# --gpu-memory-utilization: 占用 90% 显存，留 10% 给系统和通信

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name qwen3-8b-reward \
    --host 0.0.0.0 \
    --port 9000 \
    --enable-prefix-caching \
    --max-num-seqs 128 \
    --max-model-len 20000 \
    --gpu-memory-utilization 0.3 \
    --trust-remote-code