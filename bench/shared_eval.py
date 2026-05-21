import requests
import re
import logging
import os
import random # <--- 1. 新增引入 random 模块

logger = logging.getLogger(__name__)
GENERAL_JUDGE_PROMPT = """你是一个权威的法律答案评测专家。请根据提供的【问题】、【参考答案】和【AI助手的最终回答】，按照以下维度对AI助手的回答进行综合打分（0 - 100分）：

1. **核心事实覆盖 (Accuracy, 40分)**: AI的最终结论是否涵盖了参考答案中的关键法律事实或结论？
2. **证据支撑性 (Grounding, 30分)**: 结论是否建立在法律依据或类似案例之上？
3. **逻辑与完整性 (Reasoning, 30分)**: AI的推理过程是否逻辑严密，是否避免了捏造法条（幻觉）？

**评分参考**：
- 0-30分: 严重错误。结论与参考答案矛盾，或存在严重幻觉（捏造法条）。
- 31-60分: 结论基本正确，但证据无效，或逻辑混乱，或遗漏关键事实。
- 61-85分: 结论正确，证据有效，逻辑清晰。允许回答比参考答案更详细。
- 86-100分: 完美回答。证据精准，推理深刻，结论完全准确且表述专业。

【问题】: {question}
【参考答案】: {reference}
【AI助手的回答】: {ai_response}

请严格按照以下格式输出你的评测结果：
先输出简短的评分理由，最后必须在一行中输出最终得分，格式为：[[分数]]，例如 [[85]]。"""
# ================= 通用 API 调用函数 =================
def call_vllm_api(prompt, model_name, port=8009, max_tokens=2024, temperature=0.1):
    """通用的 vLLM API 调用函数"""
    # 增加 ChatML 包装，防止模型退化为文本续写
    formatted_prompt = f"<|im_start|>system\n你是一个客观、公正的 AI 评测助手。<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    url = f"http://127.0.0.1:{port}/v1/completions"
    payload = {
        "model": model_name, # 动态接收模型名称
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["<|im_end|>"]
    }
    try:
        response = requests.post(url, json=payload, timeout=200)
        response.raise_for_status()
        result_text = response.json()["choices"][0]["text"].strip()
        
        # ================= 新增：统一的 1/64 评测轨迹抽样 =================
        if random.randint(1, 64) == 1:
            logger.info(
                f"\n========== [Central Eval Trace 1/64] Model: {model_name} ==========\n"
                f"[Judge Input (Formatted)]:\n{formatted_prompt}\n\n"
                f"[Judge Output]:\n{result_text}\n"
                f"====================================================================\n"
            )
        # ==================================================================
        
        return result_text
    
    except Exception as e:
        logger.error(f"API调用失败: {e}")
        print("[debug] request payload:")
        print(payload)
        return "-1"

def parse_score_100(result_text):
    """解析 [[85]] 格式的0-100分数"""
    try:
        match = re.findall(r"\[\[\s*(\d{1,3})\s*\]\]", result_text)
        if match:
            return int(match[-1])
        return 0 # 解析失败默认给0分
    except:
        return 0
    
# ================= 新增：传统评分管道配置 =================
# 1. 动态将 lawbench_utils 加入环境变量，确保内部的 utils 包可以被顺利 import
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
lawbench_utils_dir = os.path.join(current_dir, "lawbench_utils")
if lawbench_utils_dir not in sys.path:
    sys.path.append(lawbench_utils_dir)

try:
    from evaluation_functions import (
        jec_ac, jec_kd, cjft, ydlj, ftcs, jdzy, jetq, 
        ljp_accusation, ljp_article, ljp_imprison, 
        wbfl, xxcq, flzx, wsjd, yqzy, lblj, zxfl, sjjc
    )
    
    TRADITIONAL_FUNCT_DICT = {
        "3-6": jec_ac.compute_jec_ac, "1-2": jec_kd.compute_jec_kd, "3-2": cjft.compute_cjft,
        "3-8": flzx.compute_flzx, "1-1": ftcs.compute_ftcs, "2-2": jdzy.compute_jdzy,
        "3-7": jetq.compute_jetq, "3-3": ljp_accusation.compute_ljp_accusation,
        "3-1": ljp_article.compute_ljp_article, "3-4": ljp_imprison.compute_ljp_imprison,
        "3-5": ljp_imprison.compute_ljp_imprison, "2-3": wbfl.compute_wbfl,
        "2-6": xxcq.compute_xxcq, "2-1": wsjd.compute_wsjd, "2-4": zxfl.compute_zxfl,
        "2-7": yqzy.compute_yqzy, "2-8": lblj.compute_lblj, "2-5": ydlj.compute_ydlj,
        "2-9": sjjc.compute_sjjc, "2-10": sjjc.compute_cfcy
    }
except ImportError as e:
    logger.warning(f"未能导入 traditional evaluation_functions: {e}")
    TRADITIONAL_FUNCT_DICT = {}