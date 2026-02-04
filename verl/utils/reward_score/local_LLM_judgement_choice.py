import re
import random
from openai import OpenAI

# =============================================================================
# 1. 核心 Prompt 定义：单点评估 (Pointwise Evaluation)
# =============================================================================
# 专门针对选择题 + 多轮检索场景设计。
# 强检查“证据引用”和“检索必要性”，防止模型偷懒或瞎编。
JUDGE_PROMPT_TEMPLATE = """你是一个公正的法律专家评委。请评估AI助手在解答一道法律选择题时的表现。

[用户问题]
{question}

[标准答案]
{ground_truth}

[AI助手的回答（包含思考、检索和最终结论）]
{model_output}

请根据以下三个核心维度对AI助手的表现进行打分（0.0 - 1.0分）：
1. **检索有效性**：模型是否进行了必要的检索（<search>）？检索关键词是否合理？
2. **证据一致性**：最终答案是否严格基于检索到的信息推导得出？是否存在“未检索却编造法条”的幻觉？
3. **逻辑完整性**：是否遵循了“思考-检索-再思考-结论”的严密逻辑链？

评分标准参考：
- **0.0分**：存在严重幻觉（捏造证据），或检索结果与答案完全矛盾。
- **0.1-0.4分**：进行了检索，但未能有效利用检索信息，或者逻辑链条断裂。
- **0.5-0.8分**：检索合理，逻辑清晰，但有多余步骤或轻微瑕疵。
- **0.9-1.0分**：完美的检索策略，证据引用精准，逻辑无懈可击，准确推导出答案。

请仅输出一个评分数字，格式严格为：[[分数]]，例如 [[0.8]] 或 [[1.0]]。不要输出任何其他解释。 /no_think"""


# =============================================================================
# 2. 单例客户端管理 (同步模式)
# =============================================================================
_GLOBAL_CLIENT = None

def get_client(api_url="http://localhost:9000/v1"):
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        # 强制同步调用，不使用 Async/Batch
        _GLOBAL_CLIENT = OpenAI(api_key="EMPTY", base_url=api_url)
    return _GLOBAL_CLIENT


class VLLMRewardManager:
    def __init__(self, api_url="http://localhost:9000/v1"):
        self.client = get_client(api_url)
        # 确保这里指向你部署的 Judge 模型 (推荐使用 Qwen-2.5-72B-Instruct 或类似的强推理模型)
        self.model_name = "qwen3-8b-reward" 

    def get_quality_score(self, question: str, ground_truth: str, model_output: str) -> float:
        """
        同步调用 LLM 进行打分
        """
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            model_output=model_output
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, # 贪婪采样，保证评分稳定性
                max_tokens=16,   # 只需要一个数字，限制 token 数
            )
            content = response.choices[0].message.content.strip()
            
            # 提取 [[0.x]] 格式的分数
            match = re.search(r"\[\[(\d+\.?\d*)\]\]", content)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score)) # 确保在 0-1 之间
            else:
                print(f"[Warning] Judge output format error: {content}")
                return 0.5 # 格式解析失败时的保守保底分
        except Exception as e:
            print(f"[Error] Judge API Call Failed: {e}")
            return 0.0


# =============================================================================
# 3. 辅助函数：提取与格式检查
# =============================================================================

def extract_solution(solution_str):
    """
    升级版提取函数：支持多选题（如 ABD, A,B,D 等格式）。
    策略：
    1. 寻找文本中“最后出现”的选项组合。
    2. 支持 "ABD" (紧凑) 或 "A, B, D" (分隔) 格式。
    """
    if not solution_str:
        return None
        
    # 1. 提取 <answer> 标签内的原始内容
    content_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(content_pattern, solution_str, re.DOTALL))
    if not matches:
        return None
    
    raw_content = matches[-1].group(1).strip()
    
    # 2. 定义多选正则模式
    # 逻辑：
    # (?<![a-zA-Z]) : 前面不能是字母（防止把单词中间的字母当选项）
    # [A-E]+       : 核心选项 (匹配 A 或 ABC)
    # (?: ... )* : 后续可能跟着分隔符和其他选项
    # [^a-zA-Z0-9]*: 分隔符（允许空格、逗号、&、和、顿号等非字母数字字符）
    # (?![a-zA-Z]) : 后面不能是字母
    
    # 这个正则会匹配像 "A, B", "A B D", "ABD", "A和C" 这样的块
    pattern = r"(?<![a-zA-Z])([A-E]+(?:[^a-zA-Z0-9]+[A-E]+)*)(?![a-zA-Z])"
    
    candidates = re.findall(pattern, raw_content)
    
    if candidates:
        # 取最后一个匹配块（通常是结论）
        last_match = candidates[-1]
        
        # 3. 清洗与标准化
        # 移除所有非字母字符，只保留 A-E
        clean_chars = re.sub(r"[^A-E]", "", last_match.upper())
        
        # 去重并排序，确保 "B, A" 变成 "AB"
        # set() 去重，sorted() 排序，"".join() 合并
        final_answer = "".join(sorted(set(clean_chars)))
        
        return final_answer
        
    return None

def correct_format(text):
    """
    严格的格式检查器。
    要求：必须包含思考、检索、且标签闭合正确。
    这是保护 SFT 多轮检索能力不退化的关键。
    """
    # 1. 必须包含关键标签
    required_tags = [ "<search>", "</search>", "<answer>", "</answer>"]
    for tag in required_tags:
        if tag not in text:
            return False
            
    # 2. 答案标签必须唯一 (防止模型输出多个选项迷惑判分)
    if text.count("<answer>") != 1:
        return False
        
    # 3. 必须进行至少一次检索 (针对你的"多轮检索"需求)
    if text.count("<search>") < 1:
        return False

    return True


# =============================================================================
# 4. 核心入口函数 (保持接口一致)
# =============================================================================

def compute_score(solution_str, ground_truth, extra_info=None):
    """
    RL Reward 计算主入口。
    
    Returns:
        float: 奖励分数
    """
    # ---------------- Step 1: 解析输入 ----------------
    question = extra_info.get('question', "题目内容缺失") if extra_info else "题目内容缺失"
    
    # # 鲁棒地处理 Ground Truth (支持列表或字符串)
    # reference = ground_truth
    # if isinstance(ground_truth, list) and len(ground_truth) > 0:
    #     reference = ground_truth[0]
    # reference = str(reference).strip().upper() # 转大写，方便比对
    # 假设 GT 可能是 "ABD" 或 "A,B,D" 或 list ["A","B","D"]
    reference_raw = ground_truth.get("target", [])
    if isinstance(ground_truth, list) and len(ground_truth) > 0:
        # 如果 GT 本身是 list，先转成字符串处理
        reference_raw = "".join(str(x) for x in ground_truth)
    # elif isinstance(ground_truth, dict) and 'target' in ground_truth:
    #     reference_raw = ground_truth['target']
    else:
        reference_raw = str(reference_raw)
    
    # 清洗 GT：只保留 A-E，去重并排序
    # 这样 "ABD" 和 "BDA" 都会变成 "ABD"
    reference = "".join(sorted(set(re.sub(r"[^A-E]", "", reference_raw.upper()))))

    # 提取模型预测的答案
    prediction = extract_solution(solution_str)
    if prediction:
        prediction = prediction.strip().upper()

    # ---------------- Step 2: 格式硬约束 (Format Gate) ----------------
    # 如果连答案都提不出来，或者是格式严重破损（缺检索），直接惩罚
    if prediction is None:
        return -1.0
    
    if not correct_format(solution_str):
        # 答案提取到了，但是格式不对（例如跳过了 search 直接给 answer）
        # 给予 -0.5 的惩罚。这比选错答案 (-1.0) 好一点，但比正确结构差很远。
        return -0.5 

    # ---------------- Step 3: 准确率硬约束 (Accuracy Gate) ----------------
    # 如果选项选错了，直接罚分，不需要浪费 LLM Judge
    if prediction != reference:
        # print(f"[debug] prediction:{prediction}")
        # print(f"[debug] reference:{reference}")
        return -0.2

    # ---------------- Step 4: 过程质量评估 (Quality Gate) ----------------
    # 只有当：1.格式完美 2.答案正确 时，才进入这里。
    
    manager = VLLMRewardManager()
    
    # 同步调用 Judge
    quality_score = manager.get_quality_score(
        question=question,
        ground_truth=reference,
        model_output=solution_str
    )
    
    # ---------------- Step 5: 计算最终 Reward ----------------
    # 组合逻辑：
    # Base Reward (0.5): 只要做对了题，保底拿 0.5 分。
    # Quality Bonus (0.0~0.5): 根据 LLM 评价的推理质量加分。
    # 满分 1.0，最低 0.5 (对于答对的情况)。
    
    final_score = 0.5 + (0.5 * quality_score)
    
    # ---------------- 日志采样 ----------------
    if random.randint(1, 10) == 1:
        print(f"\n[RL Log] GT:{reference} | Pred:{prediction} | Judge:{quality_score:.2f} | Final:{final_score:.2f}")

    return final_score