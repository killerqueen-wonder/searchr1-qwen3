import json
import logging
import re
import time
import requests
import os
import random
from transformers import AutoTokenizer

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

A2S_SYSTEM_PROMPT = """A conversation between User and Assistant. \
The user asks a question, and the assistant solves it. \
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. \
During thinking, the assistant can invoke the law search tool to search for law information if needed. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{{answer here}} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{{}} with latex format. \
User: {prompt}. Assistant:"""

R_SEARCH_SYSTEM_PROMPT = "You are a helpful assistant that can solve the given question step by step. For each step, start by explaining your thought process. If additional law information is needed, provide a specific query enclosed in <search> and </search>. The system will return the top search results within <observation> and </observation>. You can perform multiple searches as needed.\nWhen you know the final answer, use <original_evidence> and </original_evidence> to provide all potentially relevant original information from the observations. Ensure the information is complete and preserves the original wording without modification. If no searches were conducted or observations were made, you can skip the summary step. Finally, provide the final answer within <answer> and </answer> tags."


class VLLM_Retriever_Agent:
    def __init__(self, vllm_url, retrieve_path=None, model_name="Qwen3-8B", max_turn=12, topk=10):
        # 新增：读取 API 模式的环境变量配置
        self.use_direct_api = os.getenv("USE_DIRECT_API", "false").lower() == "true"
        self.api_key = os.getenv("API_KEY", "EMPTY")
        env_url = os.getenv("MAIN_API_URL")

        if self.use_direct_api and env_url:
            self.vllm_url = env_url
        else:
            self.vllm_url = f"{vllm_url}/v1/completions"
            
        self.retrieve_path = retrieve_path
        self.model_name = model_name
        self.max_turn = max_turn
        self.topk = topk
        self.search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

        self.tokenizer = None
        if "r-search" in self.model_name.lower() or "r_search" in self.model_name.lower():
            self.max_turn = 5  # 限定检索上限为 5 轮
            self.topk = 5      # 限定检索返回结果 top k = 5
            model_path = os.getenv("MODEL_PATH", "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/R-Search")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception as e:
                logger.error(f"加载 R-Search Tokenizer 失败: {e}")

    def _extract_tag(self, text, tag):
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1].strip() if matches else None

    def _search(self, query_str):
        if not self.retrieve_path:
            return "未启用检索功能。"
        
        # ================= JSON 鲁棒性清洗开始 =================
        clean_str = query_str.strip()
        
        # 1. 剥离可能存在的 Markdown 代码块 (例如 ```json 和 ```)
        if clean_str.startswith("```json"):
            clean_str = clean_str[7:]
        elif clean_str.startswith("```"):
            clean_str = clean_str[3:]
        if clean_str.endswith("```"):
            clean_str = clean_str[:-3]
            
        clean_str = clean_str.strip()
        
        # 2. 中文标点符号容错替换
        clean_str = clean_str.replace("，", ",")        # 中文逗号转英文逗号
        clean_str = clean_str.replace("“", '"')       # 中文左双引号转英文
        clean_str = clean_str.replace("”", '"')       # 中文右双引号转英文
        
        # 3. 结构修补：首尾缺失大括号自动补全
        if not clean_str.startswith("{"):
            clean_str = "{" + clean_str
        if not clean_str.endswith("}"):
            clean_str = clean_str + "}"
        # ================= JSON 鲁棒性清洗结束 =================

        try:
            search_query_dict = json.loads(clean_str)
        except json.JSONDecodeError as e:
            return f"工具调用失败：<search>标签内的JSON格式解析错误 ({str(e)})。请检查键值对是否正确，并严格按照JSON格式输出再试一次。"

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


# ======= 新增：提取 \boxed{} 内容的方法 (复刻老代码逻辑) =======
    def _extract_boxed(self, text):
        idx = text.rfind(r"\boxed{")
        if idx == -1:
            return text.strip()  # 兜底：如果没找到boxed，返回原文本
        
        stack = 0
        start = idx + 7
        for i in range(start, len(text)):
            if text[i] == '{':
                stack += 1
            elif text[i] == '}':
                if stack == 0:
                    return text[start:i].strip()
                stack -= 1
        return text[start:].strip()

    # ======= 新增：A2S 模型专用的 RAG 检索通道 =======
    def _search_a2s(self, query_str):
        if not self.retrieve_path:
            return "未启用检索功能。"
        
        # 适配 retrieval_server_A2S.py 的接口格式
        payload = {
            "queries": [query_str.strip()],
            "topk": self.topk,
            "return_scores": True
        }
        try:
            response = requests.post(self.retrieve_path, json=payload, timeout=120)
            response.raise_for_status()
            json_data = response.json()
            
            # 解析返回结果
            results = json_data.get("result", [[]])[0]
            if not results:
                return "No documents found."
            
            formatted_docs = []
            for doc in results:
                if isinstance(doc, dict):
                    text = doc.get("text", doc.get("contents", str(doc)))
                else:
                    text = str(doc)
                formatted_docs.append(text.strip())
                
            return "\n".join(formatted_docs)[:2000] # 截断避免爆显存
        except Exception as e:
            logger.error(f"A2S 检索系统网络或内部错误: {str(e)}")
            return f"Search failed: {str(e)}"

    # ======= 新增：A2S 专用的多轮推理 Pipeline =======
    def _gen_a2s(self, question):
        
        prompt = A2S_SYSTEM_PROMPT.format(prompt=question)
        
        sys_tool_latency, sys_rag_count = 0.0, 0
        sys_prompt_tok, sys_comp_tok = 0, 0
        first_turn_prompt_tok = 0 
        final_answer = ""

        for turn in range(self.max_turn):
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": 2048,
                "temperature": 0.0,
                "stop": ["</search>", "</answer>"]
            }
            try:
                res = requests.post(self.vllm_url, json=payload, timeout=200).json()
                if "error" in res:
                    final_answer = "API Error"
                    break
                output_text = res["choices"][0]["text"]

                usage = res.get("usage", {})
                current_prompt_tokens = usage.get("prompt_tokens", len(prompt))
                sys_prompt_tok += current_prompt_tokens
                sys_comp_tok += usage.get("completion_tokens", len(output_text))
                
                # 新增：只截取第一轮的 prompt token 作为 user input
                if turn == 0:
                    first_turn_prompt_tok = current_prompt_tokens

            except Exception as e:
                logger.error(f"vLLM API 请求异常: {e}")
                final_answer = "Error"
                break

            # 触发搜索
            if "<search>" in output_text:
                search_query = output_text.split("<search>")[-1].strip()
                t0 = time.time()
                search_result_text = self._search_a2s(search_query)
                sys_tool_latency += (time.time() - t0)
                sys_rag_count += 1
                
                # 拼接检索结果并继续推断
                prompt += f"{output_text}</search> <result> {search_result_text} </result> "
                
            # 触发回答提取
            elif "<answer>" in output_text:
                answer_block = output_text.split("<answer>")[-1].strip()
                final_answer = self._extract_boxed(answer_block)
                break
            else:
                # 异常中断或文本截断，尽力提取
                final_answer = self._extract_boxed(output_text)
                break

        
        agent_metrics = {
            "tool_latency_sec": sys_tool_latency, "rag_count": sys_rag_count,
            "user_prompt_tokens": first_turn_prompt_tok, 
            "main_total_prompt_tokens": sys_prompt_tok, "main_total_comp_tokens": sys_comp_tok
        }        
        
        # 将最终提取的答案伪装为 history 传入后续评测管道
        return final_answer, agent_metrics
    
    # ==================== 新增：R-Search 专属检索通道 ====================
    def _search_r_search(self, query_str):
        if not self.retrieve_path:
            return ""
        
        payload = {"queries": [query_str.strip()], "topk": self.topk, "return_scores": True}
        try:
            response = requests.post(self.retrieve_path, json=payload, timeout=120)
            response.raise_for_status()
            json_data = response.json()
            
            results = json_data.get("result", [[]])[0]
            if not results: return ""
            
            format_reference = ''
            for doc_item in results:
                # 兼容旧字典和新字典的嵌套格式
                content = doc_item.get('document', {}).get('contents', '')
                if not content: content = doc_item.get('document', {}).get('text', '')
                if not content: content = str(doc_item)
                
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"(Title: {title}) {text}\n"
                
            return format_reference[:2000] # 防爆显存截断
        except Exception as e:
            logger.error(f"R-Search 检索系统错误: {str(e)}")
            return ""

    # ==================== 新增：R-Search 多轮推理专线 ====================
    def _get_query_r_search(self, text):
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1]
        return None

    def _gen_r_search(self, question):
        full_question = question
        
        # 按照 main.py 逻辑拼接 Prompt
        if self.tokenizer and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": R_SEARCH_SYSTEM_PROMPT},
                 {"role": "user", "content": full_question}],
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = R_SEARCH_SYSTEM_PROMPT + '\n' + full_question

        sys_tool_latency, sys_rag_count = 0.0, 0
        sys_prompt_tok, sys_comp_tok = 0, 0
        final_answer = ""
        cnt = 0
        first_turn_prompt_tok = 0
        is_first_turn = True
        
        curr_search_template = '\n\n{output_text}<observation>{search_results}</observation>\n\n'
        stop_words = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]

        while True:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": 2048, # 生成长度保持 2048 (或者兼容的长度)
                "temperature": 0.1,
                "top_p": 0.9,
                "stop": stop_words
            }
            try:
                res = requests.post(self.vllm_url, json=payload, timeout=200).json()
                if "error" in res:
                    final_answer = "API Error"
                    break
                output_text = res["choices"][0]["text"]
                finish_reason = res["choices"][0].get("finish_reason", "")
                
                usage = res.get("usage", {})
                current_prompt_tokens = usage.get("prompt_tokens", len(prompt))
                sys_prompt_tok += current_prompt_tokens
                sys_comp_tok += usage.get("completion_tokens", len(output_text))
                
                if is_first_turn:
                    first_turn_prompt_tok = current_prompt_tokens
                    is_first_turn = False

            except Exception as e:
                logger.error(f"vLLM 请求异常: {e}")
                final_answer = "Error"
                break

            # 处理被 API 截断的 Stop Word 问题
            if finish_reason == "stop" and "<search>" in output_text and "</search>" not in output_text:
                output_text += "</search>"
                
            response = output_text.strip()
            
            # 判断是否触发搜索逻辑
            is_stop_word_hit = ("</search>" in response) or ("<search>" in response)
            
            if not is_stop_word_hit or cnt >= self.max_turn:
                # ====== 提取答案 (宽松逻辑) ======
                matches = list(re.finditer(r'<answer>(.*?)</answer>', response, re.DOTALL))
                if matches:
                    final_answer = matches[-1].group(1).strip()
                else:
                    final_answer = response # 如果没有标签，全部拿出来
                break
                
            # ====== 执行检索 ======
            tmp_query = self._get_query_r_search(response)
            if not tmp_query and "<search>" in response:
                tmp_query = response.split("<search>")[-1].replace("</search>", "")
                
            search_results = ""
            if tmp_query:
                t0 = time.time()
                search_results = self._search_r_search(tmp_query)
                sys_tool_latency += (time.time() - t0)
                sys_rag_count += 1
                
            search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
            prompt += search_text
            cnt += 1

        agent_metrics = {
            "tool_latency_sec": sys_tool_latency, "rag_count": sys_rag_count,
            "user_prompt_tokens": first_turn_prompt_tok, 
            "main_total_prompt_tokens": sys_prompt_tok, "main_total_comp_tokens": sys_comp_tok
        }   
        return final_answer, agent_metrics

    # ==============================================================================

    def gen(self, query, instruction=""):

        # ========= 核心分流：A2S 专属管道拦截 =========
        
        

        question = f"{instruction}\n{query}".strip() if instruction else query.strip()


        # ================== 纯 API 直连模式 (多模型分流适配) ==================
        if self.use_direct_api:
            # 判断是否为 luwen / zju_model 系列新模型
            if "luwen" in self.model_name.lower() :
                # --- 新增线路：适配 zju_model (luwen) ---
                formatted_prompt = f"</s>Human:{question} </s>Assistant: "
                payload = {
                    "prompt": formatted_prompt,
                    "max_tokens": 2048,
                    "temperature": 0.1,
                    "top_p": 0.75,
                    "stop": ["</s>", "Human:"] # 增加截断词，防止无限生成
                }
                fallback_prompt_len = len(formatted_prompt)

            elif "a2s" in self.model_name.lower():
                return self._gen_a2s(question)
            
            elif "r-search" in self.model_name.lower() or "r_search" in self.model_name.lower():
                return self._gen_r_search(question)
            
            # ===== 修复区：新增 Qwen3 的本地直通专线 =====
            elif "qwen" in self.model_name.lower():
                formatted_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
                payload = {
                    "model": self.model_name,
                    "prompt": formatted_prompt,
                    "max_tokens": 4000,
                    "temperature": 0.1,
                    "top_p": 0.75,
                    "stop": ["<|im_end|>"]
                }
                fallback_prompt_len = len(formatted_prompt)

            # ===== 新增：LawGPT 专属本地直通线路 =====
            elif "lawgpt" in self.model_name.lower():
                # 使用标准的 Alpaca 指令模板包裹用户问题
                formatted_prompt = (
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{question}\n\n### Response:\n"
                )
                payload = {
                    "model": self.model_name,
                    "prompt": formatted_prompt,
                    "max_tokens": 2048, # LawGPT max-len 通常较短，设为2048即可
                    "temperature": 0.1,
                    "top_p": 0.75,
                    "stop": ["</s>", "### Instruction:"]
                }
                fallback_prompt_len = len(formatted_prompt)
            # ===============================================
            
            else:
                # --- 标准 OpenAI Chat API 格式 (如 DeepSeek-v3, GPT-4) ---
                payload = {
                    "model": self.model_name, # 外部 API 通常强制要求传递模型名
                    "messages": [{"role": "user", "content": question}],
                    "max_tokens": 4000,       # 强模型可以适当给大一点
                    "temperature": 0.1,
                    "top_p": 0.75
                }
                fallback_prompt_len = len(question)

            # 修复隐患1：注入 Bearer Token 鉴权
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}" 
            }
            
            start_time = time.time()
            try:
                res = requests.post(self.vllm_url, headers=headers, json=payload, timeout=200).json()
                if "error" in res:
                    logger.error(f"API 返回错误: {res['error']}")
                    output_text = f"API Error: {res['error']}"
                    print(f"[debug] gen error ({self.model_name}), payload:")
                    print(payload)
                    prompt_tokens, comp_tokens = 0, 0
                else:
                    # 修复隐患2：标准的 Chat 接口解析路径
                    output_text = res["choices"][0]["message"]["content"] 
                    
                    # 容错处理：如果 API 不返回 usage，按字符长度估算兜底
                    usage = res.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", fallback_prompt_len)
                    comp_tokens = usage.get("completion_tokens", len(output_text))
            except Exception as e:
                logger.error(f"API 请求异常: {e}")
                output_text = "Error"
                print(f"[debug] request payload ({self.model_name}):")
                print(payload)
                
                prompt_tokens, comp_tokens = 0, 0

            # 伪造 agent_metrics 保持与 infer 脚本的兼容性
            agent_metrics = {
                "tool_latency_sec": time.time() - start_time, 
                "rag_count": 0,
                "user_prompt_tokens": prompt_tokens,
                "main_total_prompt_tokens": prompt_tokens, 
                "main_total_comp_tokens": comp_tokens
            }
            if random.randint(1, 64) == 1:
                logger.info(
                    f"\n========== [Trace Monitor 1/64] Model: {self.model_name} ==========\n"
                    f"[Prompt]:\n{payload.get('prompt', 'N/A')}\n\n"
                    f"[Output]:\n{output_text}\n"
                    f"==================================================================\n"
                )
            return output_text, agent_metrics
        
        # ==========================================================

        prompt = NEW_SYSTEM_PROMPT.format(question_text=question)
        
        cnt, search_word_before = 0, ""
        history = []
        
        sys_tool_latency, sys_rag_count = 0.0, 0
        sys_user_prompt_tokens, sys_total_prompt_tokens, sys_total_completion_tokens = 0, 0, 0 

        while True:
            payload = {
                "model": self.model_name,
                "prompt": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "max_tokens": 2500,
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
                output_text = "Error"
                print("[debug] request payload:")
                print(payload)
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

        # 先把 history 拼接好
        final_output = "\n".join(history)

        # ================= 新增：1/64 概率抽样打印轨迹 =================
        if random.randint(1, 64) == 1:
            logger.info(
                f"\n========== [Trace Monitor 1/64] Model: {self.model_name} ==========\n"
                f"[Prompt]:\n{prompt}\n\n"
                f"[Output/History]:\n{final_output}\n"
                f"==================================================================\n"
            )
        # ===============================================================

        return final_output, agent_metrics

def get_universal_vllm_summary(query, history, port, model_name="Qwen3-8B"):
    
    
    
    # ================== 新增：纯 API 模式下直接透传 ==================
    if os.getenv("USE_DIRECT_API", "false").lower() == "true":
        # 直接把 gen() 跑出来的原始回答当做 summary 返回，跳过额外请求
        return history, 0, 0
    # ================================================================

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