import sys
import os

# --- 1. 环境配置 ---
# 确保指向包含 local_LLM_judgement.py 的文件夹
reward_score_dir = "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/searchr1-qwen3/verl/utils/reward_score"
if reward_score_dir not in sys.path:
    sys.path.append(reward_score_dir)

from local_LLM_judgement import reward_score_fn

def run_real_test():
    print("🚀 启动真实 vLLM 联调测试...")
    
    # --- 2. 构造测试数据 (模拟 verl 格式) ---
    # 我们构造三个场景：一个完美的回答，一个错误的回答，一个平庸的回答
    test_batch = [
        {
            "prompt": "如何用 Python 实现斐波那契数列？",
            "reference": "可以使用递归或递推。递推法效率更高：def fib(n): a, b = 0, 1; for _ in range(n): a, b = b, a + b; return a",
            "responses": "这是一个使用递推的实现：\n```python\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```"
        },
        {
            "prompt": "太阳系最大的行星是哪一颗？",
            "reference": "太阳系最大的行星是木星 (Jupiter)。",
            "responses": "我认为太阳系最大的行星是火星，因为它看起来很红。"
        }
    ]

    # --- 3. 执行判分 ---
    try:
        print(f"正在发送请求至 vLLM (确保端口与 vLLM 启动一致)...")
        scores = reward_score_fn(test_batch)
        
        # --- 4. 结果分析 ---
        print("\n" + "="*30)
        print("📊 判分结果报告：")
        print(f"样例 1 (正确回答) 得分: {scores[0]}  (预期应接近 1.0 或 0.0)")
        print(f"样例 2 (错误回答) 得分: {scores[1]}  (预期应接近 -1.0)")
        print("="*30)
        
        if any(s != 0.0 for s in scores):
            print("✅ 联调成功！Reward Model 正在正常工作。")
        else:
            print("⚠️ 警告：所有得分均为 0.0，请检查模型输出是否符合 [[X]] 格式。")

    except Exception as e:
        print(f"❌ 联调失败！错误信息: {e}")

if __name__ == "__main__":
    run_real_test()