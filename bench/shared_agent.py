import json
import logging
import re
import time
import requests

logger = logging.getLogger(__name__)

NEW_SYSTEM_PROMPT = """你是一个严谨且专业的法律AI助手。你的任务是通过逐步思考用户请求并回答法律问题。回答必须基于事实，严禁编造法律条文或案例。

### 核心指令：
0. **第一步思考**: 识别用户请求的核心法律实体，关键事实，并且提出可能涉及的法律条文。
1. **判断是否需要检索**：你可以使用检索工具。如果问题基础且你非常有把握，也可以不使用工具直接作答。
2. **支持的检索工具**（两种）：
   - **法律检索**：需要确认某项罪名的具体刑期、适用条件、或者某一司法解释的原文时使用。
   - 不要试图一次性把所有关键词都搜完。每次最多只查找两项最相关的法律。
   - 先搜索最核心的概念。不要用缩写词，尽量用完整的，最有特点的，区别于其他法条的关键词。搜索关键词示例：“刑法 盗窃罪”、“最高人民法院关于适用〈民事诉讼法〉的解释 第501条”。
   - 如果你搜索了三次依然没有找到相关条文，请直接承认未找到，修改思考思路，搜索其他条文。不要编造内容。
   - **类案检索**：用来检索相似刑事案件的判例报告（案例库只有刑事），以预测判决结果或量刑，提高置信度。
3. **如何调用工具**：如果你决定检索，**必须**输出一个严格的 JSON 字符串，并用 `<search>` 和 `</search>` 标签包裹。
   - 调用【法律检索】示例：
     <search>
     {{
       "检索类型": "法律检索",
       "关键词": "刑法 第xxx条 盗窃罪",
       "检索目的": "找到刑法中盗窃罪的刑期判定条文"
     }}
     </search>
   - 调用【类案检索】示例：
     <search>
     {{
       "检索类型": "类案检索",
       "检索案情": "张三蒙面进入邻居家，偷走现金5000元并持刀威胁屋主。",
       "罪名": ["盗窃罪", "抢劫罪"],
       "其他情节": "自首悔过"
     }}
     </search>
4. **三段论推理**：接收到检索结果后，如果事实与法条匹配，必须使用 `<syllogism>` 标签生成三段论 JSON 分析。
   - 大前提 (Major Premise)：绝对不要重复输出法条原文，使用占位符 [法条参考 X]。
   - 小前提 (Minor Premise)：将用户平常的语言表述转化为专业的法言法语。
   - 结论 (Conclusion)：案件事实是否符合该法条，如何定罪或量刑。
5. **多轮迭代检索**：每次只输出一个 `<search>` 标签。
6. **最终回答**：收集充分后，必须将最终推理结论包裹在 `<answer>` 和 `</answer>` 标签中。

以下是需要回答的问题：{question_text}
"""

class VLLM_Retriever_Agent:
    def __init__(self, vllm_url, retrieve_path=None, model_name="Qwen3-8B", max_turn=12, topk=10):
        self.vllm_url = f"{vllm_url}/v1/completions"
        self.retrieve_path = retrieve_path
        self.model_name = model_name
        self.max_turn = max_turn
        self.topk = topk
        self.search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    def _extract_tag(self, text, tag):
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1].strip() if matches else None

    def _search(self, query_str):
        if not self.retrieve_path:
            return "未启用检索功能。"
        try:
            search_query_dict = json.loads(query_str)
        except json.JSONDecodeError as e:
            return "工具调用失败：<search>标签内的JSON格式不合法，请检查并严格按照JSON格式输出再试一次。"

        payload = {"query": search_query_dict, "topk": self.topk}
        try:
            response = requests.post(self.retrieve_path, json=payload, timeout=120)
            response.raise_for_status()
            json_data = response.json()
            if "error" in json_data: return f"检索返回错误：{json_data['error']}"
            
            req_type = json_data.get("检索类型", "")
            if req_type == "法律检索":
                results = json_data.get("result", [])
                format_reference = [f"法条参考 {idx + 1} :\n{doc.get('document', {}).get('content', '')}\n" for idx, doc in enumerate(results)]
                return "【法律检索结果】\n" + "\n".join(format_reference) if format_reference else "【法律检索结果】未找到相关的法律条文。"
            elif req_type == "类案检索":
                return f"【类案检索分析报告】\n{json_data.get('llm_summary', '未检索到匹配的类案分析结果。')}"
        except Exception as e:
            return f"检索系统网络或内部错误: {str(e)}"

    def gen(self, query, instruction=""):
        question = f"{instruction}\n{query}".strip() if instruction else query.strip()
        prompt = NEW_SYSTEM_PROMPT.format(question_text=question)
        
        cnt, search_word_before = 0, ""
        history = []
        
        sys_tool_latency, sys_rag_count = 0.0, 0
        sys_user_prompt_tokens, sys_total_prompt_tokens, sys_total_completion_tokens = 0, 0, 0 

        while True:
            payload = {
                "model": self.model_name,
                "prompt": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "max_tokens": 1500,
                "temperature": 0.3,
                "stop": ["</search>", "</answer>", "<|im_end|>"],
                "stop_token_ids": [151645, 151643]
            }
            try:
                res = requests.post(self.vllm_url, json=payload, timeout=200).json()
                output_text = res["choices"][0].get("text", "")
                usage = res.get("usage", {})
                current_prompt_tokens = usage.get("prompt_tokens", 0)
                current_completion_tokens = usage.get("completion_tokens", 0)
            except Exception as e:
                logger.error(f"vLLM 请求异常: {e}")
                break

            if cnt == 0: sys_user_prompt_tokens = current_prompt_tokens
            sys_total_prompt_tokens += current_prompt_tokens
            sys_total_completion_tokens += current_completion_tokens

            if "<search>" in output_text and "</search>" not in output_text: output_text += "</search>"; action = "search"
            elif "<answer>" in output_text and "</answer>" not in output_text: output_text += "</answer>"; action = "answer"
            else: action = "answer"

            history.append(output_text)
            instruct, search_results = '', ''

            if action == "answer" or cnt > self.max_turn: break
            elif action == "search":
                tmp_query = self._extract_tag(output_text, "search")
                if tmp_query == search_word_before:
                    instruct = "请勿重复检索。尝试直接回答或使用新关键词。"
                elif tmp_query and cnt < self.max_turn:
                    search_word_before = tmp_query
                    t0 = time.time()
                    search_results = self._search(tmp_query)
                    sys_tool_latency += (time.time() - t0)
                    sys_rag_count += 1
                elif cnt == self.max_turn:
                    instruct = "跳过检索。请立即给出最终回答，包裹在 <answer> 中。"
                else:
                    instruct = "检索格式错误。请重新思考。"

            if search_results: history.append(search_results)
            prompt += self.search_template.format(output_text=output_text, search_results=search_results) + instruct
            cnt += 1

        agent_metrics = {
            "tool_latency_sec": sys_tool_latency, "rag_count": sys_rag_count,
            "user_prompt_tokens": sys_user_prompt_tokens,
            "main_total_prompt_tokens": sys_total_prompt_tokens, "main_total_comp_tokens": sys_total_completion_tokens
        }
        return "\n".join(history), agent_metrics

def get_universal_vllm_summary(query, history, port, model_name="Qwen3-8B"):
    # 更新后的统一 Prompt
    prompt = (
        "你是一个专业、严谨的法律AI助手。请根据下方提供的【原问题】与系统的【思维链解析】，提取并整理出最终答案。\n\n"
        "【核心规则】：\n"
        "1. 请针对原问题，给出一份具备精准证据支撑、逻辑推理清晰深刻、结论完全正确且表述专业的总结性回答。\n"
        "2. 必须剔除所有 `<search>`, `<syllogism>`, `<answer>` 等内部标签及机器检索痕迹，将相关法条和类案的核心内容无缝融汇到你的最终解答中。\n"
        "3. 直接向用户呈现最终结果，不要包含“根据系统解析”等机器视角的客套话。\n\n"
        f"【原问题】：{query}\n"
        f"【思维链解析】：{history}\n\n"
        "最终答案："
    )
    url = f"http://127.0.0.1:{port}/v1/completions"
    payload = {"model": model_name, "prompt": prompt, "max_tokens": 1024, "temperature": 0.1}
    
    try:
        res = requests.post(url, json=payload, timeout=60).json()
        return res["choices"][0]["text"].strip(), res.get("usage", {}).get("prompt_tokens", 0), res.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        logger.error(f"Summary API Failed: {e}")
        return history, 0, 0