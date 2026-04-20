import re
import random
from openai import OpenAI

# =============================================================================
# 1. 核心 Prompt 定义 (GRPO 定制版)
# =============================================================================
# 修改为 0-100 百分制，强迫 Judge 给出细粒度的分数，扩大 GRPO 组内方差
JUDGE_PROMPT_TEMPLATE = """你是一位公正的法律专家评委。请评估AI助手对用户法律问题的回答质量。

[用户问题]
{question}

[参考标准答案 (仅供事实核对，不要求字数/风格一致)]
{ground_truth}

[AI助手的完整回答 (包含思考<think>、检索<search>和最终结论<answer>)]
{model_output}

请按照以下维度对AI助手的回答进行综合打分（0 - 100分）：

1. **核心事实覆盖 (Accuracy, 40分)**: AI的最终结论 (<answer>) 是否涵盖了参考答案中的关键法律事实或结论？
2. **检索支撑性 (Grounding, 30分)**: 如果AI进行了检索，结论是否建立在检索结果之上？如果无需检索直接回答，事实是否准确？
3. **逻辑与完整性 (Reasoning, 30分)**: AI的推理过程是否逻辑严密，是否避免了捏造法条（幻觉）？

**评分参考**：
- **0-30分**: 严重错误。结论与参考答案矛盾，或存在严重幻觉。
- **31-60分**: 结论基本正确，但逻辑混乱，或遗漏关键事实。
- **61-85分**: 结论正确，推理有效，逻辑清晰。
- **86-100分**: 完美回答。推理深刻，结论完全准确且表述专业。

请仅输出一个评分数字，格式严格为：[[分数]]，例如 [[85]] 或 [[100]]。不要输出任何其他解释。 /no_think"""

# =============================================================================
# 2. 单例客户端管理 (同步模式)
# =============================================================================
_GLOBAL_CLIENT = None

def get_client(api_url="http://localhost:9000/v1"):
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        # 使用同步调用，避免 RL 框架中的 event loop 冲突
        _GLOBAL_CLIENT = OpenAI(api_key="EMPTY", base_url=api_url)
    return _GLOBAL_CLIENT

class VLLMRewardManager:
    def __init__(self, api_url="http://localhost:9000/v1"):
        self.client = get_client(api_url)
        self.model_name = "qwen3-8b-reward" 

    def get_subjective_score(self, question: str, ground_truth: str, model_output: str) -> float:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            model_output=model_output
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, 
                max_tokens=10,   
            )
            content = response.choices[0].message.content.strip()
            
            match = re.search(r"\[\[(\d+\.?\d*)\]\]", content)
            if match:
                score = float(match.group(1))
                # 将 0-100 映射回 0.0 - 1.0 的尺度
                return max(0.0, min(1.0, score / 100.0))
            else:
                return 0.1  # 格式解析失败的轻微惩罚
        except Exception as e:
            print(f"[Error] Judge API Call Failed: {e}")
            return 0.0

# =============================================================================
# 3. 辅助工具：更合理的格式检查
# =============================================================================
def correct_format(text):
    """
    只检查最终输出格式是否规范。不强制要求 <search>，允许模型自主决定是否检索。
    """
    if "<answer>" not in text or "</answer>" not in text:
        return False
        
    if "<search>" in text and "</search>" not in text:
        return False
    
    if "<syllogism>" in text and "</syllogism>" not in text:
        return False

    return True

def extract_answer_content(text):
    match = re.search(r"<answer>(.*?)(</answer>|$)", text, re.DOTALL)
    return match.group(1).strip() if match else ""

# =============================================================================
# 4. 核心评分逻辑 (统一为 compute_score 接口)
# =============================================================================
def compute_score(solution_str, ground_truth, extra_info=None):
    """
    统一的接口格式：接收单个 solution_str 并返回 float 分数。
    保留了 GRPO 的零方差熔断器设计。
    """
    ground_truth = ground_truth.get("target", []) if isinstance(ground_truth, dict) else ground_truth
    if isinstance(ground_truth, list):
        reference = "\n".join([str(x) for x in ground_truth if x])
    else:
        reference = str(ground_truth)
        
    question = extra_info.get('question', "题目缺失") if extra_info else "题目缺失"

    # --- Step 1: 格式门控 ---
    # GRPO 组内相对优势：0.0 意味着它一定会在组内成为垫底，获得极大的负 Advantage
    if not correct_format(solution_str):
        return 0.0  

    # --- Step 2: 启发式门控 ---
    answer_content = extract_answer_content(solution_str)
    if len(answer_content) < 5:
        return 0.05  

    # --- Step 3: 同步 LLM Judge ---
    manager = VLLMRewardManager()
    quality_score = manager.get_subjective_score(
        question=question,
        ground_truth=reference,
        model_output=solution_str
    )
    
    # --- Step 4: GRPO 零方差熔断器 (Tie-breaker) ---
    # 加上一段极其微小的基于回答长度的扰动。
    # 即使 Judge 给这几个回答都打了 0.85 分，加上扰动后就会变成 0.85012, 0.85015...
    # 从而保证标准差不为 0，模型能够区分出“在同样得分下，稍微详尽一点的更好”，保持梯度流动。
    length_bonus = min(0.01, len(answer_content) * 0.00001)
    final_score = quality_score + length_bonus
    
    # --- 日志采样 ---
    if random.randint(1, 64) == 1:
        print(f"\n[Subjective RL] Score: {final_score:.2f}")
        print(f"Q: {question[:100]}...")
        print(f"GT: {reference[:500]}...")
        # 截断长文本避免刷屏
        print(f"Model Answer: {answer_content}")

    return final_score