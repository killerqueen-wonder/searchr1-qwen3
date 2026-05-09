import re
import random
from openai import OpenAI
import time

# =============================================================================
# 1. 核心 Prompt 定义 (GRPO 定制版)
# =============================================================================
# 修改为 0-100 百分制，强迫 Judge 给出细粒度的分数，扩大 GRPO 组内方差
JUDGE_PROMPT_TEMPLATE = """你是一位公正的法律专家评委。请评估AI助手对用户法律问题的回答质量。

[用户问题]
{question}

[参考标准答案 (仅供事实核对，参考它的检索路径和逻辑推理方式，不要求字数/风格一致)]
{ground_truth}

[AI助手的完整回答 (包含思考<think>、检索<search>和最终结论<answer>)]
{model_output}

请按照以下维度对AI助手的回答进行综合打分（0 - 100分）：

1. **核心事实覆盖 (Accuracy, 40分)**: AI的最终结论 (<answer>) 是否涵盖了参考答案中的关键法律事实或结论？
2. **检索支撑性 (Grounding, 30分)**: 如果问题复杂到需要检索，AI是否进行了检索?结论是否建立在检索结果之上？
3. **逻辑与完整性 (Reasoning, 30分)**: AI的推理过程是否逻辑严密，是否避免了捏造法条（幻觉）？

**评分参考**：
- **0-30分**: 严重错误。结论与参考答案矛盾，或存在严重幻觉（捏造法条），或未进行检索直接瞎编。
- **31-60分**: 结论基本正确，但检索过程无效，或逻辑混乱，或遗漏关键事实。
- **61-85分**: 结论正确，检索有效，逻辑清晰。允许回答比参考答案更详细。
- **86-100分**: 完美回答。检索精准，推理深刻，结论完全准确且表述专业。

请仅输出一个评分数字，格式严格为：[[分数]]，例如 [[85]] 或 [[100]]。不要输出任何其他解释。 /no_think"""


# =============================================================================
# 1. 核心 Prompt 定义 (GRPO 多维度合并定制版)
# =============================================================================
UNIFIED_JUDGE_PROMPT_TEMPLATE = """你是一位严谨的AI推理行为与法律逻辑审计员。
请对比【参考数据】和【被测模型输出】，按照以下三个维度对被测模型进行综合打分，每项满分均为 100 分。

[问题]
{question}

[参考轨迹 (Ground Truth CoT，包含 thought, search, information 等标签)]
{reference_cot}

[参考答案 (Ground Truth Answer)]
{reference_answer}

[被测模型完整输出]
{model_output}

[被测模型实际搜到的内容提取]
{retrieved_info}

**评分维度与阶梯标准：**

1. **核心事实准确度 (accuracy)**:对比【被测模型完整输出】的结论部分和【参考答案】
   - [85-100分]: 最终结论完全准确，涵盖了参考答案中的关键事实。
   - [60-84分]: 结论基本正确，但遗漏了部分非核心细节或存在微小瑕疵。
   - [30-59分]: 结论部分错误，或逻辑混乱。
   - [0-29分]: 严重错误，结论与参考矛盾，或存在幻觉编造。

2. **轨迹与时机对齐 (alignment)**:对比【被测模型完整输出】的检索轨迹和【参考轨迹】
   - [85-100分]: 在与参考轨迹相同的逻辑断点处发起了检索（<search>），检索策略与参考高度一致。
   - [60-84分]: 检索时机略有偏差，或策略略宽泛，但整体逻辑合理。
   - [30-59分]: 错过了必要的检索节点，或进行了多余的低效检索。
   - [0-29分]: 参考进行了检索但模型完全未检索，或在极度不合理的地方滥用工具。

3. **信息增量价值 (info_gain)**:对比【被测模型实际搜到的内容提取】和【参考答案】
   - [85-100分]: 搜回的信息（包含在 <information> 中）极其精准，直接支撑了最终参考答案的核心内容。
   - [60-84分]: 搜回的信息部分相关，能提供背景支持，但缺失最关键的一击。
   - [30-59分]: 搜到的多为边缘信息，对解题帮助有限。
   - [0-29分]: 搜回的内容完全无关，或模型根本没有进行有效检索（无返回内容）。

请仅输出一个 JSON 字典，不要包含任何其他分析过程文字，确保包含 "accuracy", "alignment", "info_gain" 三个键。
例如：{{"accuracy": 85, "alignment": 90, "info_gain": 80}}  /no_think"""

import json

def extract_answer_content(text: str) -> str:
    """
    提取文本中【最后一次】出现 <answer> 和 </answer> 之间的内容。
    """
    # 找到所有匹配项
    matches = re.findall(r"<answer>(.*?)(?:</answer>|$)", text, re.DOTALL)
    if matches:
        # 返回最后一个匹配项并去空格
        return matches[-1].strip()
    return ""

def extract_answer_content(text):
    match = re.search(r"<answer>(.*?)(</answer>|$)", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def count_search_actions(text: str) -> int:
    """统计文本中 <search> 标签出现的次数"""
    if not text or not isinstance(text, str):
        return 0
    return len(re.findall(r"<search>", text))

def extract_information_blocks(text: str) -> str:
    """提取文本中所有的 <information> 内容供评委参考"""
    matches = re.findall(r"<information>(.*?)</information>", text, re.DOTALL)
    return "\n---\n".join(matches) if matches else "无检索内容"

def calculate_query_quality_score(solution_str: str) -> float:
    """
    仅针对“法律检索”提取相似度分数。
    忽略“类案检索”，归一化基于 0-20.0 的量程。
    """
    pattern = r"<information>.*?\[Score:\s*([\d\.]+),\s*Type:\s*法律检索\].*?</information>"
    matches = re.findall(pattern, solution_str, re.DOTALL)
    if not matches:
        return 0.0
    scores_100 = [max(0.0, min(100.0, (float(s) / 20.0) * 100.0)) for s in matches]
    return sum(scores_100) / len(scores_100)

import re
import json

def parse_judge_json(raw_str: str) -> dict:
    """鲁棒性 JSON 解析（去除 <think> 块，提取核心字典，保留符号容错）"""
    # 1. 移除 <think>...</think> 块（使用 re.DOTALL 匹配多行内容）
    clean_str = re.sub(r"<think>.*?</think>", "", raw_str, flags=re.DOTALL)
    
    # 清理可能残留的孤立标签和首尾空白
    clean_str = clean_str.replace("<think>", "").replace("</think>", "").strip()
    
    # 2. 中英文标点符号容错替换
    clean_str = clean_str.replace("，", ",").replace("“", '"').replace("”", '"').replace("：", ":")
    
    # 3. 锁定真正的 JSON 字典边界 (寻找第一个 { 和最后一个 })
    start_idx = clean_str.find('{')
    end_idx = clean_str.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        # 精准截取字典部分
        clean_str = clean_str[start_idx:end_idx+1]
    else:
        # 结构修补：首尾缺失大括号自动补全 (兜底)
        if not clean_str.startswith("{"): clean_str = "{" + clean_str
        if not clean_str.endswith("}"): clean_str = clean_str + "}"
        
    # 4. 加载验证
    try:
        data = json.loads(clean_str)
        # 确保必备键均存在
        if all(k in data for k in ["accuracy", "alignment", "info_gain"]):
            return data
    except Exception as e:
        # 如果需要，这里可以解除注释打印具体的解析错误
        print(f"[JSON Decode Error] {e} -> String: {clean_str}")
        return None
        
    return None
    
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

    def get_unified_subjective_scores(self, question: str, reference_cot: str, reference_answer: str, model_output: str, retrieved_info: str) -> dict:
        """
        调用 LLM 获取三维打分 JSON，最多重试 3 次，超时或错误则返回 0 分字典。
        """
        prompt = UNIFIED_JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            reference_cot=reference_cot,
            reference_answer=reference_answer,
            model_output=model_output,
            retrieved_info=retrieved_info
        )
        
        default_scores = {"accuracy": 0.0, "alignment": 0.0, "info_gain": 0.0}
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # 给一点微小温度防止一直采样到同样的死循环语法错误
                    max_tokens=64,
                    timeout=30.0      # 设置超时机制
                )
                content = response.choices[0].message.content.strip()
                parsed_data = parse_judge_json(content)
                
                if parsed_data is not None:
                    return {
                        "accuracy": float(parsed_data["accuracy"]),
                        "alignment": float(parsed_data["alignment"]),
                        "info_gain": float(parsed_data["info_gain"])
                    }
                else:
                    # 修复点：显式捕获解析失败，并打印原文以便 Debug
                    print(f"[Warning] Judge JSON解析失败 (尝试 {attempt+1}/{max_retries}). 模型原文输出: {content}")
                    time.sleep(1)
                    continue
                    
            except Exception as e:
                # 包含超时 (Timeout)、API 连接错误、解析异常等
                print(f"[Warning] Judge API 错误或解析失败 (尝试 {attempt+1}/{max_retries}): {e}")
                time.sleep(1) # 短暂等待后重试
                continue
                
        print("[Error] Judge API 3次尝试均失败，给予 0 分惩罚。")
        return default_scores

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



def count_search_actions(text: str) -> int:
    """
    统计文本中 <search> 标签出现的次数，以此衡量模型的检索频率。
    """
    if not text or not isinstance(text, str):
        return 0
    return len(re.findall(r"<search>", text))

# =============================================================================
# 4. 核心评分逻辑 (统一为 compute_score 接口)
# =============================================================================
# def compute_score(solution_str, ground_truth, extra_info=None):
#     """
#     统一的接口格式：接收单个 solution_str 并返回 float 分数。
#     保留了 GRPO 的零方差熔断器设计。
#     """
#     ground_truth = ground_truth.get("target", []) if isinstance(ground_truth, dict) else ground_truth
#     if isinstance(ground_truth, list):
#         reference = "\n".join([str(x) for x in ground_truth if x])
#     else:
#         reference = str(ground_truth)
        
#     question = extra_info.get('question', "题目缺失") if extra_info else "题目缺失"

#     # --- Step 1: 格式门控 ---
#     # GRPO 组内相对优势：0.0 意味着它一定会在组内成为垫底，获得极大的负 Advantage
#     if not correct_format(solution_str):
#         return 0.0  

#     # --- Step 2: 启发式门控 ---
#     answer_content = extract_answer_content(solution_str)
#     if len(answer_content) < 3:
#         return 0.05  

#     # --- Step 3: 同步 LLM Judge ---
#     manager = VLLMRewardManager()
#     quality_score = manager.get_subjective_score(
#         question=question,
#         ground_truth=reference,
#         model_output=solution_str
#     )

#     # --- Step 4: 梯度检索奖励 (Tiered Search Alignment Reward) ---
#     gt_search_count = count_search_actions(reference)
#     model_search_count = count_search_actions(solution_str)
#     diff = abs(gt_search_count - model_search_count)
    
#     search_bonus = 0.0
#     if diff <= 2:
#         # 梯度 1: 极佳对齐 (差值 <= 2)
#         search_bonus = 0.10  
#     elif diff <= 3:
#         # 梯度 2: 较好对齐 (差值 == 3)
#         search_bonus = 0.05
#     # 差值 > 3 则无 bonus
    
#     # --- Step 4: GRPO 零方差熔断器 (Tie-breaker) ---
#     # 加上一段极其微小的基于回答长度的扰动。
#     # 即使 Judge 给这几个回答都打了 0.85 分，加上扰动后就会变成 0.85012, 0.85015...
#     # 从而保证标准差不为 0，模型能够区分出“在同样得分下，稍微详尽一点的更好”，保持梯度流动。
#     length_bonus = min(0.1, len(answer_content) * 0.00001)
#     # 综合得分：基础分 + 阶梯奖金 - 长度惩罚
#     final_score = quality_score + search_bonus - length_bonus
    
#     # --- 日志采样 ---
#     if random.randint(1, 64) == 1:
#         print(f"\n[Subjective RL] Score: {final_score:.2f}")
#         print(f"Q: {question[-100:]}...")
#         print(f"GT: {reference[-500:]}...")
#         print(f"Model Answer: {answer_content}")

#     return final_score

def compute_score(solution_str, ground_truth, extra_info=None):
    """
    最终 Reward 函数实现：
    1. 主观四维加权 (Acc 0.45, Align 0.20, Info 0.20, Query 0.15)
    2. 检索次数对齐奖金 (search_bonus)
    3. 长度奖金 (length_bonus, 上限 0.1)
    """
    ground_truth = ground_truth.get("target", []) if isinstance(ground_truth, dict) else ground_truth
    if isinstance(ground_truth, list):
        reference = "\n".join([str(x) for x in ground_truth if x])
    else:
        reference = str(ground_truth)
    # 获取参考 CoT（包含 thought, search, information 等）
    reference_cot = reference
    # 提取参考答案：使用 extract_answer_content 获取最后一次 <answer>
    reference_answer = extract_answer_content(reference_cot)
    
    
    question = extra_info.get('question', "题目缺失") if extra_info else "题目缺失"

    # --- Step 1: 格式与内容基础门控 ---
    if not correct_format(solution_str):
        return 0.0  
    
    answer_content = extract_answer_content(solution_str)
    if len(answer_content) < 3:
        return 0.05  

    # --- Step 2: 检索词质量 (法律检索相似度) ---
    query_quality_100 = calculate_query_quality_score(solution_str)

    # --- Step 3: LLM Judge 多维度打分 (带重试与解析) ---
    retrieved_info = extract_information_blocks(solution_str)
    manager = VLLMRewardManager()
    subjective_scores = manager.get_unified_subjective_scores(
        question=question,
        reference_cot=reference_cot,
        reference_answer=reference_answer,
        model_output=solution_str,
        retrieved_info=retrieved_info
    )

    acc_100 = subjective_scores["accuracy"]
    align_100 = subjective_scores["alignment"]
    info_100 = subjective_scores["info_gain"]

    # --- Step 4: 启发式奖金计算 (search_bonus & length_bonus) ---
    # 检索次数对齐奖金
    gt_search_count = count_search_actions(reference_cot)
    model_search_count = count_search_actions(solution_str)
    diff = abs(gt_search_count - model_search_count)
    
    search_bonus = 0.0
    if diff <= 1:
        search_bonus = 0.10  
    elif diff <= 2:
        search_bonus = 0.05
        
    # 长度奖金项 (新公式：上限 0.1)
    length_punish = min(0.1, len(solution_str) * 0.000001)

    # --- Step 5: 最终分数聚合 ---
    # 比例：Acc(0.45) + Align(0.20) + Info(0.20) + Query(0.15)
    subjective_total = (acc_100 * 0.8 / 100.0) + \
                       (align_100 * 0.08 / 100.0) + \
                       (info_100 * 0.10 / 100.0) + \
                       (query_quality_100 * 0.02 / 100.0)

    final_score = subjective_total + search_bonus - length_punish

    # --- 日志采样 ---
    if random.randint(1, 64) == 1:
        print(f"\n[GRPO RL Reward] Final: {final_score:.4f}")
        print(f"Components -> Acc:{acc_100}, Align:{align_100}, Info:{info_100}, Query:{query_quality_100:.1f}")
        print(f"Bonuses    -> SearchBonus:{search_bonus}, LengthBonus:{length_punish:.4f}")
        print(f"Q: ...{question[-100:]}")
        print(f"GT: ...{ground_truth[-200:]}")
        print(f"Model think: {solution_str[2000:]}...")
        print(f"Model Answer: {answer_content}...")
        

    return final_score