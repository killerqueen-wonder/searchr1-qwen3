import os
import re
import json
import time
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

# ================= 默认配置区 =================
# 文件路径
INPUT_FILE = "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/dataset/reward_model_training/rollout_trajectories_json_3k_0517.jsonl"
BACKUP_DIR = "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/dataset/reward_model_training/reward_data_construction_bk_0517" # 断点续传备份目录
FINAL_TRAIN_FILE = "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/dataset/reward_model_training/train.json"
FINAL_TEST_FILE = "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/dataset/reward_model_training/test.json"

# API 全局变量 (将在 main 函数中被动态赋值)
API_KEY = None
API_BASE_URL = None
MODEL_NAME = "deepseek-v3" # DeepSeek-V3 的标准调用名称

# 运行参数
MAX_WORKERS = 30
TEST_MODE_LIMIT = None # 设为 None 跑全量，设为数字只跑前 N 条测试
MAX_TEXT_LENGTH = 40000 # 单次请求最大字符数安全限制

# ================= 裁判 Prompt 模板 =================
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

# ================= 文本处理函数 =================
def extract_answer_content(text: str) -> str:
    match = re.search(r".*<answer>(.*?)</answer>", text, re.DOTALL)
    if match: return match.group(1).strip()
    last_index = text.rfind("<answer>")
    if last_index != -1: return text[last_index + len("<answer>"):].strip()
    return "未提取到有效回答"

def extract_information_blocks(text: str) -> str:
    matches = re.findall(r"<information>(.*?)</information>", str(text), re.DOTALL)
    return "\n---\n".join(matches) if matches else "未进行有效检索"

def truncate_text(text: str, max_len: int = 15000) -> str:
    """对超长文本进行截断，保留头部和尾部"""
    if not text or len(text) <= max_len:
        return text
    half = max_len // 2
    return text[:half] + "\n\n...[由于长度限制，中间部分已截断]...\n\n" + text[-half:]

# ================= API 调用逻辑 =================
def call_judge_api(prompt: str) -> str:
    """调用大模型 API 获取打分结果，带 3 次重试机制"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1, 
        "max_tokens": 100
    }
    
    for attempt in range(3):
        try:
            response = requests.post(API_BASE_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result_text = response.json()['choices'][0]['message']['content'].strip()
            
            # 验证是否为合法 JSON
            json.loads(result_text)
            return result_text
            
        except Exception as e:
            time.sleep(2 * (attempt + 1))
            if attempt == 2:
                print(f"[API 错误] 连续3次请求失败: {e}")
                return '{"accuracy": 1, "alignment": 1, "info_gain": 1}'

# ================= 核心处理逻辑 =================
def process_single_item(item: Dict) -> None:
    """处理单条数据并保存到备份"""
    item_id = item.get("id")
    backup_file = os.path.join(BACKUP_DIR, f"{item_id}.json")
    
    # 检查是否已处理过 (断点续传)
    if os.path.exists(backup_file):
        return
        
    # 提取并组装变量
    question = item.get("original_question", "")
    ref_cot = item.get("reference_cot", "无参考轨迹")
    ref_answer = item.get("original_golden_answer", "无参考答案")
    model_output = item.get("generated_cot", "")
    
    ans_content = extract_answer_content(model_output)
    retrieved_info = extract_information_blocks(model_output)
    
    model_output = truncate_text(model_output, MAX_TEXT_LENGTH // 4)
    retrieved_info = truncate_text(retrieved_info, MAX_TEXT_LENGTH // 4)
    ref_cot = truncate_text(ref_cot, MAX_TEXT_LENGTH // 4)
    
    final_prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        reference_cot=ref_cot,
        reference_answer=ref_answer,
        model_output=model_output,
        retrieved_info=retrieved_info,
        answer_content=ans_content
    )
    
    # 调用 API
    score_json_str = call_judge_api(final_prompt)
    
    # 构建 SFT 目标字段
    complex_cot_content = f"{final_prompt}\n\n{score_json_str}\n{score_json_str}\n{score_json_str}"
    
    final_dict = {
        "id": item_id,
        "format_id": item.get("format_id", ""),
        "Open-ended Verifiable Question": "", 
        "Complex_CoT": complex_cot_content
    }
    
    with open(backup_file, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, ensure_ascii=False)
    print(f"[SUCCESS] 数据 ID {item_id} 审计完成并保存。")

# ================= 主流程 =================
def main():
    global API_KEY, API_BASE_URL, MODEL_NAME
    
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="合成数据轨迹自动化打分脚本")
    parser.add_argument("--api-key", type=str, help="大模型 API Key。也可通过环境变量 JUDGE_API_KEY 传入。")
    parser.add_argument("--api-url", type=str, help="大模型 API Base URL。也可通过环境变量 JUDGE_API_URL 传入。")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="调用的模型名称，默认: deepseek-chat")
    args = parser.parse_args()
    
    # --- 安全赋值逻辑 (参数优先，环境变量兜底) ---
    API_KEY = args.api_key or os.getenv("JUDGE_API_KEY")
    API_BASE_URL = args.api_url or os.getenv("JUDGE_API_URL")
    MODEL_NAME = args.model
    
    if not API_KEY or not API_BASE_URL:
        print("[ERROR] 启动失败：未检测到 API 凭据！")
        print("请使用以下两种方式之一提供 API 信息：")
        print("1. 命令行传参: python judge_pipeline.py --api-key \"sk-xxx\" --api-url \"https://...\"")
        print("2. 环境变量法: export JUDGE_API_KEY=\"sk-xxx\" && export JUDGE_API_URL=\"https://...\"")
        return

    # --- 开始执行业务逻辑 ---
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        
    print(f"[INFO] 正在使用的模型: {MODEL_NAME}")
    print(f"[INFO] 开始加载 Rollout 数据: {INPUT_FILE}")
    
    data_list = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))
                
    if TEST_MODE_LIMIT:
        data_list = data_list[:TEST_MODE_LIMIT]
        print(f"[INFO] 处于测试模式，仅处理前 {TEST_MODE_LIMIT} 条数据。")

    print(f"[INFO] 启动线程池打分 (最大并发数: {MAX_WORKERS}) ...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_item, item) for item in data_list]
        for future in as_completed(futures):
            try:
                future.result() # 如果子线程报错，这里会强制在主终端打印堆栈
            except Exception as e:
                print(f"[线程崩溃] 错误详情: {e}")
    # 临时替换为：
    # print("[DEBUG] 正在使用单线程顺序执行，以便捕获任何潜在报错...")
    # for item in data_list:
    #     process_single_item(item)

    print("\n[INFO] 所有数据打分完毕，开始合并与拆分数据集...")
    
    merged_data = []
    for filename in os.listdir(BACKUP_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(BACKUP_DIR, filename), "r", encoding="utf-8") as f:
                merged_data.append(json.load(f))
                
    merged_data.sort(key=lambda x: str(x.get("id", "")))
    total_count = len(merged_data)
    print(f"[INFO] 成功收集到 {total_count} 条已处理数据。")
    
    if total_count > 300:
        test_data = merged_data[:100]
        train_data = merged_data[100:]
        
        with open(FINAL_TEST_FILE, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        with open(FINAL_TRAIN_FILE, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
            
        print(f"[INFO] 已保存 {len(test_data)} 条至 {FINAL_TEST_FILE}，{len(train_data)} 条至 {FINAL_TRAIN_FILE}。")
    else:
        with open(FINAL_TRAIN_FILE, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 已全部保存 {total_count} 条至 {FINAL_TRAIN_FILE}。")
        
    print("[INFO] 管道脚本执行完毕！")

if __name__ == "__main__":
    main()