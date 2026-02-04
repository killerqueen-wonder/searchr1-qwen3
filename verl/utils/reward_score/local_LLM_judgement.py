import re
import random
from openai import OpenAI

# =============================================================================
# 1. 核心 Prompt 定义：针对主观题 + 多轮检索的单点评估
# =============================================================================
# 关键改进：
# 1. 明确告知 Judge 标准答案(Reference)可能很简短，而模型输出(Output)包含思考过程。
# 2. 评分维度聚焦于：事实准确性(Recall)、检索支撑性(Grounding)、逻辑性(Reasoning)。
JUDGE_PROMPT_TEMPLATE = """你是一位公正的法律专家评委。请评估AI助手对用户法律问题的回答质量。

[用户问题]
{question}

[参考标准答案 (仅供事实核对，不要求字数/风格一致)]
{ground_truth}

[AI助手的完整回答 (包含思考<think>、检索<search>和最终结论<answer>)]
{model_output}

请按照以下标准对AI助手的回答进行打分（0.0 - 1.0分）：

1. **核心事实覆盖 (Accuracy)**: AI的最终结论 (<answer>) 是否涵盖了参考答案中的关键法律事实或结论？
2. **检索支撑性 (Grounding)**: AI是否进行了有效的检索 (<search>)？其最终结论是否建立在检索结果之上，而非凭空捏造？
3. **逻辑与完整性 (Reasoning)**: AI的推理过程是否逻辑严密？

**评分指南**：
- **0.0 - 0.3**: 严重错误。结论与参考答案矛盾，或存在严重幻觉（捏造法条），或未进行检索直接瞎编。
- **0.4 - 0.6**: 结论基本正确，但检索过程无效，或逻辑混乱，或遗漏关键事实。
- **0.7 - 0.9**: 结论正确，检索有效，逻辑清晰。允许回答比参考答案更详细。
- **1.0**: 完美回答。检索精准，推理深刻，结论完全准确且表述专业。

请仅输出一个评分数字，格式严格为：[[分数]]，例如 [[0.8]] 或 [[1.0]]。不要输出任何其他解释。 /no_think"""


# =============================================================================
# 2. 单例客户端管理 (同步模式)
# =============================================================================
_GLOBAL_CLIENT = None

def get_client(api_url="http://localhost:9000/v1"):
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        # 必须使用同步调用，避免 RL 框架中的 event loop 冲突
        _GLOBAL_CLIENT = OpenAI(api_key="EMPTY", base_url=api_url)
    return _GLOBAL_CLIENT


class VLLMRewardManager:
    def __init__(self, api_url="http://localhost:9000/v1"):
        self.client = get_client(api_url)
        # 建议使用推理能力较强的模型作为 Judge (如 Qwen-2.5-72B 或 GPT-4o-mini)
        self.model_name = "qwen3-8b-reward" 

    def get_subjective_score(self, question: str, ground_truth: str, model_output: str) -> float:
        """
        同步调用 LLM 进行主观题打分
        """
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            model_output=model_output # 注意：这里传入的是全量输出！
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, # 贪婪采样确保评分一致性
                max_tokens=10,   # 只需要一个数字
            )
            content = response.choices[0].message.content.strip()
            
            # 解析 [[0.x]] 格式
            match = re.search(r"\[\[(\d+\.?\d*)\]\]", content)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            else:
                print(f"[Warning] Judge output format error: {content}")
                return 0.5 # 格式解析失败时的保底分
        except Exception as e:
            print(f"[Error] Judge API Call Failed: {e}")
            return 0.0


# =============================================================================
# 3. 辅助工具：格式检查与内容提取
# =============================================================================

def correct_format(text):
    """
    格式硬约束：必须包含思考、检索标签，且闭合正确。
    这对于保持 SFT 5.5 的多轮检索能力至关重要。
    """
    # 1. 关键标签检查
    required_tags = [ "<search>", "</search>", "<answer>", "</answer>"]
    for tag in required_tags:
        if tag not in text:
            return False
            
    # 2. 结构检查：必须有至少一次检索
    if text.count("<search>") < 1:
        return False
        
    # 3. 结构检查：Answer 标签只能出现一次（防止输出多个结论）
    if text.count("<answer>") > 1:
        # 允许 retry 或多次 answer 的情况比较少见，通常限制为 1 次以保证简洁
        return False

    return True

def extract_answer_content(text):
    """提取最终结论用于日志记录（可选），Judge 看的是全文"""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else "No Answer Found"


# =============================================================================
# 4. 核心入口函数 (Compute Score)
# =============================================================================

def compute_score(solution_str, ground_truth, extra_info=None):
    """
    针对 test.parquet 主观题数据集的打分逻辑。
    """
    # --- Step 1: 解析输入数据 ---
    # parquet 中的 golden_answers 是一个 list，通常包含参考文本
    # 我们将其拼接成字符串，作为 Reference 提供给 Judge
    if isinstance(ground_truth, list):
        reference = "\n".join([str(x) for x in ground_truth if x])
    else:
        reference = str(ground_truth)
    
    question = extra_info.get('question', "题目内容缺失") if extra_info else "题目内容缺失"

    # --- Step 2: 格式硬约束 (Format Gate) ---
    # 格式不对直接罚分，不浪费 LLM 调用
    if not correct_format(solution_str):
        # 格式错误给 -0.5，比内容完全错误稍微好一点点（鼓励模型至少尝试生成），
        # 或者给 -1.0 强力惩罚。这里建议 -0.5。
        return -0.5

    # --- Step 3: 长度/内容基础检查 (Heuristic Gate) ---
    # 防止模型输出空内容的 <answer></answer>
    answer_content = extract_answer_content(solution_str)
    if len(answer_content) < 5:
        return -0.8 # 回答太短，几乎肯定是错的

    # --- Step 4: LLM Judge 深度评估 ---
    # 主观题必须过 Judge，无法像选择题那样做正则匹配
    manager = VLLMRewardManager()
    
    # 传入全量 solution_str (包含 search 和 think)，解决“Reward模型眼瞎”的问题
    quality_score = manager.get_subjective_score(
        question=question,
        ground_truth=reference,
        model_output=solution_str
    )
    
    # --- Step 5: 计算最终 Reward ---
    # 线性映射：Score (0.0~1.0) -> Reward
    # 你可以根据需要调整系数。
    # 示例：满分 1.0，最低 0.0。
    final_score = quality_score
    
    # --- 日志采样 ---
    if random.randint(1, 32) == 1:
        print(f"\n[Subjective RL] Score: {final_score:.2f}")
        print(f"Q: {question[:50]}...")
        print(f"GT: {reference[:50]}...")
        print(f"Model Answer: {answer_content[:50]}...")

    return final_score