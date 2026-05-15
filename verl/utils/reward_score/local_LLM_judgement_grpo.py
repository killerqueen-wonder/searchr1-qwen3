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

def parse_single_score(raw_str: str) -> float:
    """从模型输出中提取 [[85]] 格式的分数"""
    clean_str = re.sub(r"<think>.*?</think>", "", raw_str, flags=re.DOTALL)
    match = re.search(r"\[\[(\d+)\]\]", clean_str)
    if match:
        return float(match.group(1))
    # 兼容模型偶尔只输出纯数字的情况
    nums = re.findall(r"\d+", clean_str)
    if nums:
        return float(nums[0])
    return 0.0

def correct_format(text):
    #answer必须齐全，search和syllogism必须闭合
    if "<answer>" not in text or "</answer>" not in text: return False
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
# 3. 客户端与并发请求管理
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

    def _call_single_dimension(self, prompt_template: str, kwargs: dict, dimension_name: str) -> float:
        """负责单次 API 调用，处理超长截断、重试与异常屏蔽"""
        max_retries = 3
        # 初始赋值
        current_kwargs = kwargs.copy()
        
        for attempt in range(max_retries):
            prompt = prompt_template.format(**current_kwargs)
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=32, # 只需输出数字，降低输出负担
                    timeout=300.0 
                )
                content = response.choices[0].message.content.strip()
                score = parse_single_score(content)
                return score
                
            except BadRequestError as e:
                # 捕获超长报错: HTTP 400
                error_msg = str(e).lower()
                if "maximum context length" in error_msg or "context" in error_msg:
                    print(f"[Warning] {dimension_name} API 触发 Context Limit (尝试 {attempt+1}/{max_retries}). 尝试截断文本...")
                    
                    # 进行字符串级别截断（保留末尾，因为结论往往在后面）
                    # 约保留最后 40000 个字符 (~25000 tokens)，留下充足余量
                    max_chars = 35000 
                    if "model_output" in current_kwargs and len(current_kwargs["model_output"]) > max_chars:
                        current_kwargs["model_output"] = "...[前文已截断]..." + current_kwargs["model_output"][-max_chars:]
                    elif "reference_cot" in current_kwargs and len(current_kwargs["reference_cot"]) > max_chars:
                        current_kwargs["reference_cot"] = "...[前文已截断]..." + current_kwargs["reference_cot"][-max_chars:]
                    else:
                        # 如果没有长字段可截断，直接跳出重试
                        break 
                    continue # 截断后立刻进行下一次尝试
                else:
                    print(f"[Warning] {dimension_name} 出现其他 400 错误: {e}")
                    time.sleep(1)
                    continue
                    
            except Exception as e:
                # 其他 API 异常 (连接错误、超时等)，独立隔离，不影响其他维度
                print(f"[Warning] {dimension_name} API 未知异常 (尝试 {attempt+1}/{max_retries}): {e}")
                time.sleep(1)
                
        print(f"[Error] {dimension_name} 最终获取失败，当前维度返回 0 分。")
        return 0.0

    def get_unified_subjective_scores(self, reference_cot: str, reference_answer: str, retrieved_info: str, answer_content: str, model_output: str) -> dict:
        """使用线程池并发获取三个维度的分数"""
        
        # 准备任务参数
        tasks = {
            "accuracy": (JUDGE_PROMPT_ACCURACY, {"reference_answer": reference_answer, "answer_content": answer_content}),
            "alignment": (JUDGE_PROMPT_ALIGNMENT, {"reference_cot": reference_cot, "model_output": model_output}),
            "info_gain": (JUDGE_PROMPT_INFO_GAIN, {"reference_answer": reference_answer, "retrieved_info": retrieved_info}),
        }

        results = {"accuracy": 0.0, "alignment": 0.0, "info_gain": 0.0}
        
        # 开启 3 个并发线程，压榨 vLLM 服务器性能
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_dim = {
                executor.submit(self._call_single_dimension, prompt_tpl, kwargs, dim): dim 
                for dim, (prompt_tpl, kwargs) in tasks.items()
            }
            
            for future in concurrent.futures.as_completed(future_to_dim):
                dim = future_to_dim[future]
                try:
                    results[dim] = future.result()
                except Exception as exc:
                    print(f"[Error] 线程执行维度 {dim} 时崩溃: {exc}")
                    results[dim] = 0.0
                    
        return results

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
    if not correct_format(solution_str):
        print("[debug]warning: 模型输出标签格式错误。")
        final_score -= 0.2 

    if check_search_json(solution_str):
        #检索格式是否正确
        final_score += 0.02

    if check_information_tags_strict(solution_str):
        #检索结果是否成功返回
        final_score += 0.08 
    
    answer_content = extract_answer_content(solution_str)

    # --- Step 2: 检索词质量 ---
    query_quality_100 = calculate_query_quality_score(solution_str)

    # --- Step 3: LLM Judge 多维度打分 (并发获取) ---
    retrieved_info = extract_information_blocks(solution_str)
    manager = VLLMRewardManager()
    
    # 传入完整的 solution_str 作为 model_output 以备截断使用
    subjective_scores = manager.get_unified_subjective_scores(
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
    if diff <= 1: search_bonus = 0.01  
        
    length_punish_limit = 0.2
    length_punish = min(length_punish_limit, (len(solution_str)*0.000005)*length_punish_limit)

    # --- Step 5: 最终分数聚合 ---
    subjective_total = (acc_4 * 1 / 4.0) + \
                       (align_4 * 0.02 / 4.0) + \
                       (info_4 * 0.08 / 4.0) + \
                       (query_quality_100 * 0.05 / 100.0)

    final_score = final_score + subjective_total + search_bonus - length_punish

    if random.randint(1, 32) == 1:
        print(f"\n[GRPO RL Reward] Final: {final_score:.4f}")
        print(f"Components -> Acc:{acc_4}, Align:{align_4}, Info:{info_4}, Query:{query_quality_100:.1f}")
        print(f"Bonuses    -> search_bonus:{search_bonus:.4f},LengthBonus:{length_punish:.4f}")
        print(f"Q: ...{question[-200:]}")
        print(f"GT: ...{reference[-2000:]}")
        print(f"Model think: {solution_str}...")
        print(f"Model Answer: {answer_content}...")
         
    return final_score