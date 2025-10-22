#!/usr/bin/env python
# -*- coding: utf-8 -*-

import transformers
import torch
import re
import requests
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessor,
    LogitsProcessorList
)
import os

# 设置环境变量，避免某些库的警告
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 示例问题
question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"
question = 'who got the first nobel prize in physics?'
# Model ID and device setup
model_id = "/linzhihang/huaiwenpang/legal_LLM/Search-R1/Search-R1/verl_checkpoints/nq-search-r1-ppo-llama3.2-3b-em/actor/global_step_1000"
# model_id ='meta-llama/Llama-3.2-3B'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 确保问题以问号结尾
question = question.strip()
if question[-1] != '?':
    question += '?'

# 搜索结果模板
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# 准备 prompt（保留原始示例，但稍后会处理答案提取问题）
prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you lack knowledge, call a search engine via <search> query </search>. \
It will return results between <information> and </information>. \
You can search multiple times. \
If no more knowledge is needed, provide the answer in <answer> content </answer>, e.g., <answer> Beijing </answer>. \
Question: {question}\n"""

# 记录原始 prompt 长度，用于后续区分生成内容
original_prompt_length = len(prompt)

# 初始化 tokenizer 和模型
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

# 设置 pad_token（Llama-3 默认没有 pad_token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id
print(f'[debug] eos_token_id={eos_token_id}, pad_token_id={pad_token_id}')


def get_query(text):
    """从文本中提取最后一次 <search>...</search> 中的查询"""
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    else:
        return None


def clean_generated_text(text):
    """清理生成的文本，移除无意义内容"""
    # 移除代码片段模式
    text = re.sub(r'def\s+\w+\s*\(.*?\)\s*:', '', text)
    # 移除重复的特殊字符序列
    text = re.sub(r'(sup\d{8}\s*){2,}', '', text)
    # 移除过多的数字序列
    text = re.sub(r'\d{4,}\s*', '', text)
    return text.strip()


def search(query: str):
    """调用本地搜索引擎 API"""
    if not query or not query.strip():
        print("[WARNING] Empty query passed to search function.")
        return ""

    payload = {
        "queries": [query],
        "topk": 3,
        "return_scores": True
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8006/retrieve",
            json=payload,
            proxies={"http": None, "https": None},
            timeout=10
        )
        response.raise_for_status()
        json_data = response.json()
        results = json_data.get('result', [])
    except requests.exceptions.Timeout:
        print("[ERROR] Search request timed out.")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        return ""
    except ValueError as e:
        print(f"[ERROR] Failed to decode JSON: {e}")
        print("Response content:", response.text[:500])
        return ""

    if not results:
        print("[INFO] No results returned from search.")
        return ""

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            lines = content.split("\n")
            title = lines[0] if lines else ""
            text = "\n".join(lines[1:]) if len(lines) > 1 else ""
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


class AvoidEOSTokenProcessor(LogitsProcessor):
    """阻止模型生成 EOS token"""
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id
        
    def __call__(self, input_ids, scores):
        scores[:, self.eos_token_id] = float("-inf")
        return scores


class RobustStopOnTag(StoppingCriteria):
    """基于 token ID 匹配的停止条件，解决分词断裂问题"""
    def __init__(self, tokenizer, target_tag):
        self.target_ids = tokenizer.encode(target_tag, add_special_tokens=False)
        self.buffer = []
        
    def __call__(self, input_ids, scores, **kwargs):
        new_token = input_ids[0, -1].item()
        self.buffer.append(new_token)
        
        if len(self.buffer) >= len(self.target_ids):
            if self.buffer[-len(self.target_ids):] == self.target_ids:
                self.buffer = []  # 重置 buffer
                return True
        return False
    
    def reset(self):
        """重置 buffer，用于下一次生成"""
        self.buffer = []


# 开始推理循环
print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
print(f'[prompt]: {prompt}')

final_answer = None
max_search_attempts = 3
search_attempt = 0

while search_attempt < max_search_attempts:
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 每次生成都新建 stopping_criteria，避免 buffer 累积
    stopping_criteria = StoppingCriteriaList([
        RobustStopOnTag(tokenizer, "</search>"),
        RobustStopOnTag(tokenizer, "</answer>")
    ])

    # 创建 LogitsProcessor
    logits_processor = LogitsProcessorList([
        AvoidEOSTokenProcessor(tokenizer.eos_token_id)
    ])

    # 生成：禁用 eos_token_id 停止机制
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        stopping_criteria=stopping_criteria,
        pad_token_id=pad_token_id,
        do_sample=True,
        temperature=0.3,  # 降低温度增加确定性
        top_p=0.9,
        repetition_penalty=1.5,  # 增加重复惩罚
        no_repeat_ngram_size=3,  # 防止 3-gram 重复
        logits_processor=logits_processor,
        eos_token_id=None,  # 关键：不要因生成 EOS 而提前退出
    )

    # 解码新增内容（仅新生成的部分）
    generated_tokens = outputs[0][input_ids.shape[1]:]
    new_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    new_text = clean_generated_text(new_text)
    
    # 更新完整 prompt
    prompt += new_text
    search_attempt += 1

    print(f'[Newly generated content in NO.{search_attempt}]:\n{new_text}')
    
    # 只在新生成的内容中搜索 <search> 和 <answer>
    # 这是关键修复：避免匹配到原始 prompt 中的示例
    search_query = get_query(new_text)
    
    if search_query:
        print(f"[INFO] Detected search query: {search_query}")
        search_results = search(search_query)
        if not search_results.strip():
            search_results = "No relevant information found."
        
        # 构造搜索结果文本
        search_text = curr_search_template.format(output_text=new_text, search_results=search_results)
        prompt += search_text
        print(f'[Added search results NO.{search_attempt}]:\n{search_text}')
    else:
        # 检查新生成的内容中是否有答案
        answer_match = re.search(r"<answer>(.*?)</answer>", new_text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            final_answer = answer_match.group(1).strip()
            print(f"\n✅ Found answer in newly generated content: <answer> {final_answer} </answer>")
            break
        else:
            # 没有搜索也没有答案，继续下一轮
            print("[INFO] No search query or answer found in this generation. Continuing...")

# 如果循环结束仍未找到答案，尝试在最后生成的内容中提取
if not final_answer:
    # 再次检查最后生成的内容
    last_generated = prompt[original_prompt_length:]
    answer_match = re.search(r"<answer>(.*?)</answer>", last_generated, re.DOTALL | re.IGNORECASE)
    if answer_match:
        final_answer = answer_match.group(1).strip()
    

# 输出最终结果
if final_answer:
    print(f"\n🎯 Final Answer: <answer> {final_answer} </answer>")
else:
    print("\n❌ Could not determine the answer after maximum search attempts.")
    print("Last generated content may contain clues:")
    print(prompt[original_prompt_length:][-500:])

print("\n################# [Reasoning + Searching Finished] ##################")