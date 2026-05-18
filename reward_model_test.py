import os
import json
import re
import numpy as np
from vllm import LLM, SamplingParams

# ================= 配置区 =================
# 模型与数据路径 (请根据实际情况修改)
MODEL_PATH = "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/SFT_ckp/qwen3_reward_model_0517/checkpoint-2-2172/tfmr"
TEST_JSON_PATH = "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/dataset/reward_model_training/test.json"
OUTPUT_RESULT_PATH = "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/dataset/reward_model_training/eval_results_compare.json"

# vLLM 参数
NUM_GPUS = 2 # 评测阶段 1 张 H20 足够，若要加速可改为 2 或 4
MAX_MODEL_LEN = 40000

# ================= 1. 文本分离与解析函数 =================
def split_rating_text(text: str) -> tuple[str, str]:
    """
    从文本中匹配最后一次出现的“请输出你的评分：”，
    将其及之前的内容保存为 q，之后的内容保存为 a。
    """
    # 兼容性处理：如果你之前的 prompt 用的是 "=== 输出要求 ===" 或者其他，可以在这里增加 fallback
    target = "请输出你的评分："
    
    last_index = text.rfind(target)
    
    if last_index == -1:
        # 如果没有找到目标子字符串，尝试用常规的JSON左括号作为备选切割点
        alt_index = text.rfind('{"accuracy"')
        if alt_index != -1:
            return text[:alt_index], text[alt_index:]
        return text, ""
    
    split_point = last_index + len(target)
    q = text[:split_point]
    a = text[split_point:]
    
    return q, a

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

# ================= 2. 主评测流程 =================
def main():
    print(f"[INFO] 正在加载测试数据: {TEST_JSON_PATH}")
    with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
        # 兼容 JSON list 和 JSONL 两种格式
        content = f.read().strip()
        if content.startswith('['):
            data_list = json.loads(content)
        else:
            data_list = [json.loads(line) for line in content.split('\n') if line]

    print(f"[INFO] 成功加载 {len(data_list)} 条数据。")

    print(f"[INFO] 正在初始化 vLLM 引擎: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH, 
        tensor_parallel_size=NUM_GPUS, 
        trust_remote_code=True,
        enable_prefix_caching=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.85 
    )
    tokenizer = llm.get_tokenizer()
    
    # 采样参数 (评测任务使用 temperature=0.0 以获得确定性输出)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=200, # 仅输出分数 JSON，200 足矣
        stop=['<|im_end|>']
    )

    prompts_to_infer = []
    ground_truths = []
    
    # 提取 Prompt 并进行 Chat Template 包装
    for item in data_list:
        complex_cot = item.get("Complex_CoT", "")
        q, a = split_rating_text(complex_cot)
        
        # 提取真实打分 (Ground Truth)
        gt_scores = parse_scores(a)
        ground_truths.append(gt_scores)
        
        # 核心：使用 Chat Template 包装 q (加上 <|im_start|>user... 等)
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts_to_infer.append(formatted_prompt)

    # 批量推理
    print(f"[INFO] 开始推理 (共 {len(prompts_to_infer)} 条)...")
    outputs = llm.generate(prompts_to_infer, sampling_params, use_tqdm=True)

    # ================= 3. 对比与统计 =================
    exact_match_count = 0        # 3项全对的数据条数
    item_match_count = 0         # 匹配的单项总数
    total_items = len(data_list) * 3  # 总单项数 (数据条数 * 3)
    
    all_diffs = []               # 存放所有的分数绝对差值
    
    results_record = []          # 详细结果记录
    
    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text
        pred_scores = parse_scores(gen_text)
        gt_scores = ground_truths[i]
        
        # 比较单项差异
        acc_diff = abs(pred_scores["accuracy"] - gt_scores["accuracy"])
        align_diff = abs(pred_scores["alignment"] - gt_scores["alignment"])
        info_diff = abs(pred_scores["info_gain"] - gt_scores["info_gain"])
        
        all_diffs.extend([acc_diff, align_diff, info_diff])
        
        # 单项匹配统计
        current_item_matches = 0
        if acc_diff == 0: current_item_matches += 1
        if align_diff == 0: current_item_matches += 1
        if info_diff == 0: current_item_matches += 1
        
        item_match_count += current_item_matches
        
        # 三项全对统计
        if current_item_matches == 3:
            exact_match_count += 1
            
        results_record.append({
            "id": data_list[i].get("id", i),
            "ground_truth": gt_scores,
            "model_predict": pred_scores,
            "raw_output": gen_text
        })

    # ================= 4. 打印最终结果 =================
    total_samples = len(data_list)
    
    # 软标准统计
    diff_array = np.array(all_diffs)
    mean_diff = np.mean(diff_array)
    std_diff = np.std(diff_array)
    
    print("\n" + "="*50)
    print("                SFT 裁判模型评测报告")
    print("="*50)
    
    print(f"📊 【硬标准 - 整体一致性】")
    print(f"   完全一致(3项全对)比例: {exact_match_count}/{total_samples} ({(exact_match_count/total_samples)*100:.2f}%)")
    
    print(f"\n📊 【中等标准 - 单项一致性】")
    print(f"   单项分数一致比例: {item_match_count}/{total_items} ({(item_match_count/total_items)*100:.2f}%)")
    
    print(f"\n📊 【软标准 - 分数偏差度量】")
    print(f"   平均绝对差值 (MAE): {mean_diff:.4f} 分 (越接近0越好)")
    print(f"   分差标准差 (Std)  : {std_diff:.4f}")
    print("="*50)

    # 保存详细结果，供后期 Review 分析
    with open(OUTPUT_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results_record, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 详细预测结果已保存至: {OUTPUT_RESULT_PATH}")

if __name__ == "__main__":
    main()