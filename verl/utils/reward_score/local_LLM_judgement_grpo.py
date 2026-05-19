import re
import json
import time
import random
import concurrent.futures
from openai import OpenAI, BadRequestError

# =============================================================================
# 1. 拆分版 Prompt 定义 (精准裁剪上下文，缩小输入长度)
# =============================================================================

JUDGE_PROMPT_ACCURACY = """你是一位严谨的法律事实审计员。请对比【参考答案】和【被测模型最终回答】，给出核心事实准确度打分 (1-4分)。

[参考答案]
{reference_answer}
[参考答案结束]

[被测模型最终回答]
{answer_content}
[被测模型最终回答结束]

评分标准：
- [4分]: 最终结论完全准确，与参考答案关键事实一致。
- [3分]: 结论基本正确，但遗漏部分细节或带微小错误。
- [2分]: 结论部分错误。
- [1分]: 严重错误，结论矛盾，回答为空或出现无意义字符。

请仅输出一个评分数字，格式严格为：[[分数]]，例如 [[3]]。不要输出其他解释。 /no_think"""


JUDGE_PROMPT_ALIGNMENT = """你是一位AI推理行为审计员。请对比【参考轨迹】和【被测模型完整轨迹】，根据被测轨迹的检索时机与参考轨迹是否对齐，给出打分 (1-4分)。

[参考轨迹]
{reference_cot}
[参考轨迹结束]

[被测模型完整轨迹]
{model_output}
[被测模型完整轨迹结束]

评分标准：
- [4分]: 检索时机和策略与参考轨迹高度一致。
- [3分]: 检索时机或策略略有偏差，但整体合理。
- [2分]: 错过必要检索节点，或进行了多余的低效检索。
- [1分]: 该检索却未检索，或极度滥用工具。

请仅输出一个评分数字，格式严格为：[[分数]]，例如 [[2]]。不要输出其他解释。 /no_think"""


JUDGE_PROMPT_INFO_GAIN = """你是一位信息价值审计员。请对比【被测模型的检索内容】和【参考答案】，判断被测模型的检索内容是否与参考答案相关，给出信息增量价值打分 (1-4分)。

[参考答案]
{reference_answer}
[参考答案结束]

[被测模型的检索内容]
{retrieved_info}
[被测模型的检索内容结束]

评分标准：
- [4分]: 搜回信息极精准，直接支撑参考答案的核心内容。
- [3分]: 信息部分相关，提供背景支持但非关键一击。
- [2分]: 搜到多为边缘信息，帮助有限。
- [1分]: 完全无关，或未进行有效检索(无返回)。

请仅输出一个评分数字，格式严格为：[[分数]]，例如 [[3]]。不要输出其他解释。 /no_think"""

JUDGE_PROMPT_TEMPLATE = """你是一位严谨的 AI 行为与法律事实审计员。请根据提供的【问题】、【参考轨迹】、【参考答案】，对【被测模型完整轨迹】、【被测模型检索内容】、【被测模型最终回答】进行三个维度的严格打分 (1-4分)。
【问题】是用户提出的问题，【参考轨迹】是参考模型对问题的COT过程，【参考答案】是参考模型给出的答案，【被测模型完整轨迹】是被测模型对问题的COT过程，【被测模型检索内容】是被测模型在思考问题时检索的内容，【被测模型最终回答】是被测模型给出的答案。
你需要针对被测模型的三项输出内容，分别给出三项分数。以下提供所有信息，以及三项评分标准。
=== 输入信息 ===
[问题]
{question}

[参考轨迹]
{reference_cot}

[参考答案]
{reference_answer}

[被测模型完整轨迹]
{model_output}

[被测模型检索内容]
{retrieved_info}

[被测模型最终回答]
{answer_content}
=== 信息结束 ===

=== 评分标准 ===
1. 核心事实准确度 (accuracy): 对比【参考答案】和【被测模型最终回答】。
   - [4分]: 最终结论完全准确，与参考答案关键事实一致。
   - [3分]: 结论基本正确，但遗漏部分细节或带微小错误。
   - [2分]: 结论部分错误。
   - [1分]: 严重错误，结论矛盾，回答为空或出现无意义字符。

2. 检索策略与对齐度 (alignment): 对比【参考轨迹】和【被测模型完整轨迹】。
   - [4分]: 检索时机和策略与参考轨迹高度一致。
   - [3分]: 检索时机或策略略有偏差，但整体合理。
   - [2分]: 错过必要检索节点，或进行了多余的低效检索。
   - [1分]: 该检索却未检索，或极度滥用工具。

3. 信息增量价值 (info_gain): 对比【被测模型的检索内容】和【参考答案】。
   - [4分]: 搜回信息极精准，直接支撑参考答案的核心内容。
   - [3分]: 信息部分相关，提供背景支持但非关键一击。
   - [2分]: 搜到多为边缘信息，帮助有限。
   - [1分]: 完全无关，或未进行有效检索(无返回)。

=== 输出要求 ===
请仔细思考以上三个维度的表现，最后**仅输出**一个合法的 JSON 对象，不要输出任何其他的解释文字、Markdown 代码块或思考过程标记。格式必须严格如下：
{{"accuracy": 0, "alignment": 0, "info_gain": 0}}
请输出你的评分："""

# =============================================================================
# 2. 辅助函数 (保持不变)
# =============================================================================
def extract_answer_content(text):
    match = re.search(r".*<answer>(.*?)</answer>", text, re.DOTALL)
    if match: return match.group(1).strip()
    last_index = text.rfind("<answer>")
    if last_index != -1: return text[last_index + len("<answer>"):].strip()
    return ""

def count_search_actions(text: str) -> int:
    return len(re.findall(r"<search>", str(text)))

def extract_information_blocks(text: str) -> str:
    matches = re.findall(r"<information>(.*?)</information>", str(text), re.DOTALL)
    return "\n---\n".join(matches) if matches else "无检索内容"

def calculate_query_quality_score(solution_str: str) -> float:
    pattern = r"<information>.*?\[Score:\s*([\d\.]+),\s*Type:\s*法律检索\].*?</information>"
    matches = re.findall(pattern, str(solution_str), re.DOTALL)
    if not matches: return 0.0
    scores_100 = [max(0.0, min(100.0, (float(s) / 20.0) * 100.0)) for s in matches]
    return sum(scores_100) / len(scores_100)

# def parse_single_score(raw_str: str) -> float:
#     """从模型输出中提取 [[85]] 格式的分数"""
#     clean_str = re.sub(r"<think>.*?</think>", "", raw_str, flags=re.DOTALL)
#     match = re.search(r"\[\[(\d+)\]\]", clean_str)
#     if match:
#         return float(match.group(1))
#     # 兼容模型偶尔只输出纯数字的情况
#     nums = re.findall(r"\d+", clean_str)
#     if nums:
#         return float(nums[0])
#     return 0.0
def parse_scores(text: str) -> dict:
    """
    高度鲁棒的分数提取函数：
    1. 读取第一个 { 到第一个 }
    2. 去掉空格，中文引号、冒号、逗号全部改为英文
    3. 尝试 JSON 解析
    4. 如果失败，利用关键词正则匹配，匹配失败的项记为 0 分。
    """
    default_scores = {"accuracy": 0, "alignment": 0, "info_gain": 0}
    if not text:
        return default_scores

    start = text.find('{')
    end = text.find('}')
    
    if start != -1 and end != -1 and end > start:
        # 1. 截取字典字符串
        json_str = text[start:end+1]
        
        # 2. 清洗：去除空白字符（包括空格、换行）
        json_str = re.sub(r'\s+', '', json_str)
        # 3. 清洗：中文标点替换
        json_str = json_str.replace("“", '"').replace("”", '"').replace("：", ":").replace("，", ",")
        
        try:
            # 4. 尝试解析为合法的 JSON
            data = json.loads(json_str)
            return {
                "accuracy": int(data.get("accuracy", 0)),
                "alignment": int(data.get("alignment", 0)),
                "info_gain": int(data.get("info_gain", 0))
            }
        except Exception:
            pass # 如果抛出 JSONDecodeError 等异常，放行到下方的正则 fallback

    # 5. 正则 Fallback 匹配
    # 匹配模式解释：寻找关键词，中间允许有任意非数字字符，然后捕获第一个出现的数字
    for key in default_scores.keys():
        # 尝试匹配带引号的 "accuracy": 3
        match = re.search(rf'"{key}"[^0-9]*([0-9])', text, re.IGNORECASE)
        if not match:
            # 尝试匹配无引号的 accuracy: 3
            match = re.search(rf'{key}[^0-9]*([0-9])', text, re.IGNORECASE)
        
        if match:
            default_scores[key] = int(match.group(1))

    return default_scores

def correct_format(text):
    #answer必须齐全，search和syllogism必须闭合
    
    # 按顺序找出所有的 <answer> 和 </answer> 标签
    tags = re.findall(r'(<answer>|</answer>)', text)
    
    # 1. 如果标签总数小于 2，或者不是偶数（成对出现），直接返回 False
    if len(tags) < 2 or len(tags) % 2 != 0:
        return False
    
    # 2. 检查是否交替出现
    # 偶数索引（0, 2, 4...）必须是 '<answer>'
    # 奇数索引（1, 3, 5...）必须是 '</answer>'
    for i, tag in enumerate(tags):
        if i % 2 == 0 and tag != '<answer>':
            return False
        if i % 1 == 0 and i % 2 != 0 and tag != '</answer>':
            return False
        
    if "<search>" in text and "</search>" not in text: return False
    if "<syllogism>" in text and "</syllogism>" not in text: return False
    return True

def check_information_tags_strict(text: str) -> bool:
    """
    判断字符串中是否至少存在一对 <information> 标签，
    并且 <information> 和 </information> 必须严格交替出现。
    """
    # 使用正则表达式提取所有开始和结束标签，按出现顺序存入列表
    # </?information> 会匹配 <information> 和 </information>
    tags = re.findall(r'</?information>', text)
    
    # 条件1：至少要存在标签（且如果是正确交替，总数必定是偶数，至少2个）
    if not tags:
        return False
        
    # 状态变量：0 表示当前等待开始标签，1 表示当前等待结束标签
    state = 0 
    
    for tag in tags:
        if tag == "<information>":
            if state == 1:
                # 错误：上一个标签也是开始标签（出现了 <info> <info>）
                return False
            state = 1
            
        elif tag == "</information>":
            if state == 0:
                # 错误：没有开始标签就直接结束，或者连续出现了结束标签（如 </info> 或 <info></info></info>）
                return False
            state = 0
            
    # 条件2：遍历结束后，状态必须回到 0，即所有打开的标签都已正确闭合
    return state == 0

def check_search_json(text: str) -> bool:
    """
    判断字符串中是否至少存在一对 <search> 和 </search>，
    且其中包含符合特定格式要求的 JSON 文本。
    """
    # 使用正则表达式提取 <search> 和 </search> 之间的所有内容
    # re.DOTALL 参数允许 '.' 匹配包括换行符在内的任意字符
    matches = re.findall(r'<search>(.*?)</search>', text, re.DOTALL)
    
    for content in matches:
        try:
            # 尝试将提取到的内容解析为 JSON 字典
            # json.loads 会自动忽略首尾的空白字符和换行
            data = json.loads(content)
            
            # 确保解析出来的是一个 JSON 对象（字典），而不是数组或普通字符串
            if not isinstance(data, dict):
                continue
            
            # 将 JSON 的所有键提取为集合，方便进行精确比对
            keys = set(data.keys())
            
            # 校验规则 1：法律检索
            if data.get("检索类型") == "法律检索":
                expected_keys = {"检索类型", "关键词", "检索目的"}
                # 集合比对：不仅要求包含这三个键，且不能有其他多余的键
                if keys == expected_keys:
                    return True
                    
            # 校验规则 2：类案检索
            elif data.get("检索类型") == "类案检索":
                expected_keys = {"检索类型", "检索案情", "罪名", "其他情节"}
                # 集合比对：必须且只能包含这四个键
                if keys == expected_keys:
                    return True
                    
        except json.JSONDecodeError:
            # 如果 json.loads 抛出异常，说明标签内的内容不是合法的 JSON
            # 捕获异常并跳过，继续检查下一对标签
            continue
            
    # 如果遍历完所有匹配项都没有符合条件的，则返回 False
    return False

# =============================================================================
# 3. 客户端与请求管理 (重构版)
# =============================================================================
_GLOBAL_CLIENT = None

def get_client(api_url="http://localhost:9000/v1"):
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        _GLOBAL_CLIENT = OpenAI(api_key="EMPTY", base_url=api_url)
    return _GLOBAL_CLIENT

class VLLMRewardManager:
    def __init__(self, api_url="http://localhost:9000/v1"):
        self.client = get_client(api_url)
        self.model_name = "qwen3-8b-reward" 

    def get_unified_subjective_scores(self, question: str, reference_cot: str, reference_answer: str, retrieved_info: str, answer_content: str, model_output: str) -> dict:
        """单次 API 调用获取三个维度的分数，处理超长截断、重试与异常屏蔽"""
        max_retries = 3
        
        # 初始化入参字典
        kwargs = {
            "question": question,
            "reference_cot": reference_cot,
            "reference_answer": reference_answer,
            "model_output": model_output,
            "retrieved_info": retrieved_info,
            "answer_content": answer_content
        }
        
        for attempt in range(max_retries):
            prompt = JUDGE_PROMPT_TEMPLATE.format(**kwargs)
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=128, # 由于需要输出完整JSON格式，放宽 token 限制
                    timeout=300.0 
                )
                content = response.choices[0].message.content.strip()
                scores = parse_scores(content)
                return scores
                
            except BadRequestError as e:
                # 捕获超长报错: HTTP 400
                error_msg = str(e).lower()
                if "maximum context length" in error_msg or "context" in error_msg:
                    print(f"[Warning] 统一评分 API 触发 Context Limit (尝试 {attempt+1}/{max_retries}). 尝试截断文本...")
                    
                    # 组合后总长度变大，单字段截断长度应更保守，约保留 15000 字符 (~10000 tokens)
                    max_chars = 15000 
                    needs_retry = False
                    if len(kwargs["model_output"]) > max_chars:
                        kwargs["model_output"] = "...[前文已截断]..." + kwargs["model_output"][-max_chars:]
                        needs_retry = True
                    if len(kwargs["reference_cot"]) > max_chars:
                        kwargs["reference_cot"] = "...[前文已截断]..." + kwargs["reference_cot"][-max_chars:]
                        needs_retry = True
                        
                    if not needs_retry:
                        break # 如果没有长字段可截断，直接跳出重试
                    continue 
                else:
                    print(f"[Warning] 统一评分 API 出现其他 400 错误: {e}")
                    time.sleep(1)
                    continue
                    
            except Exception as e:
                # 其他 API 异常 (连接错误、超时等)
                print(f"[Warning] 统一评分 API 未知异常 (尝试 {attempt+1}/{max_retries}): {e}")
                time.sleep(1)
                
        print(f"[Error] 统一评分最终获取失败，当前返回全 0 分。")
        return {"accuracy": 0, "alignment": 0, "info_gain": 0}
    
# =============================================================================
# 4. 核心评分逻辑 
# =============================================================================
def compute_score(solution_str, ground_truth, extra_info=None):
    
    ground_truth = ground_truth.get("target", []) if isinstance(ground_truth, dict) else ground_truth
    if isinstance(ground_truth, list):
        reference = "\n".join([str(x) for x in ground_truth if x])
    else:
        reference = str(ground_truth)
        
    reference_cot = reference
    reference_answer = extract_answer_content(reference_cot)
    question = extra_info.get('question', "题目缺失") if extra_info else "题目缺失"

    final_score = 0
    # --- Step 1: 格式与内容基础门控 ---
    if correct_format(solution_str):
        final_score += 0.50  # 取代原来的 -0.2

    if check_search_json(solution_str):
        final_score += 0.25  # 原为 +0.1

    if check_information_tags_strict(solution_str):
        final_score += 0.25  # 原为 +0.1
    
    answer_content = extract_answer_content(solution_str)
    

    # --- Step 2: 检索词质量 ---
    query_quality_100 = calculate_query_quality_score(solution_str)

    # --- Step 3: LLM Judge 多维度打分 (合并获取) ---
    retrieved_info = extract_information_blocks(solution_str)
    manager = VLLMRewardManager()
    
    # 传入需要的所有参数
    subjective_scores = manager.get_unified_subjective_scores(
        question=question,
        reference_cot=reference_cot,
        reference_answer=reference_answer,
        retrieved_info=retrieved_info,
        answer_content=answer_content,
        model_output=solution_str
    )

    acc_4 = subjective_scores["accuracy"]
    align_4 = subjective_scores["alignment"]
    info_4 = subjective_scores["info_gain"]

    # --- Step 4: 启发式奖金计算 ---
    gt_search_count = count_search_actions(reference_cot)
    model_search_count = count_search_actions(solution_str)
    diff = abs(gt_search_count - model_search_count)
    
    search_bonus = 0.0
    #与参考轨迹检索次数相近
    if diff <= 1: search_bonus = 0.1  
    #简单题，参考轨迹与被测轨迹都无需检索，补偿可能的检索信息失分
    if gt_search_count == 0 and model_search_count==0 : search_bonus = 0.2
        
    length_punish_limit = 0.6
    length_punish = min(length_punish_limit, (len(solution_str)*0.000005)*length_punish_limit)

    if answer_content=='和' or answer_content=='':
        #不输出答案，直接判负
        acc_4 = 0.1

    # --- Step 5: 最终分数聚合 ---
    subjective_total = (acc_4 * 1.00 / 4.0) + \
                       (align_4 * 0.08 / 4.0) + \
                       (info_4 * 0.32 / 4.0) + \
                       (query_quality_100 * 0.20 / 100.0)
    
    

    final_score = final_score + subjective_total + search_bonus - length_punish

    if random.randint(1, 32) == 1:
        print("=="*20)
        print(f"\n[GRPO RL Reward] Final: {final_score:.4f}")
        print(f"Components -> Acc:{acc_4}, Align:{align_4}, Info:{info_4}, Query:{query_quality_100:.1f}")
        print(f"Bonuses    -> search_bonus:{search_bonus:.4f},LengthBonus:{length_punish:.4f}")
        print(f"Q: ...{question[-200:]}")
        print(f"GT: ...{reference[-1000:]}")
        print(f"Model think: {solution_str}...")
        print(f"Model Answer: {answer_content}...")
        print("=="*20)
         
    return final_score