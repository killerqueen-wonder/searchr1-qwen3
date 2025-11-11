import os
import re
import torch
import requests
import transformers
from transformers.generation.utils import GenerationConfig


class StopOnSequence(transformers.StoppingCriteria):
    """用于检测指定token序列（如 </search> ）的生成停止条件"""
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in target_sequences]
        self.target_lengths = [len(t) for t in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for t, l in zip(self.target_ids, self.target_lengths):
            target = torch.as_tensor(t, device=input_ids.device)
            if torch.equal(input_ids[0, -l:], target):
                return True
        return False


class LLM_retriever:
    """
    实现一个“思考-检索-再思考-回答”的闭环生成系统。
    """
    def __init__(self, model_path, model_name=None, api_key=None, api_url=None, retrieve_path="http://127.0.0.1:8006/retrieve"):
        self.model_path = model_path
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
        self.retrieve_path = retrieve_path

        print("--------------加载模型路径为：---------------\n", model_path)

        # 初始化模型与tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # 特殊token设置
        if 'Qwen' in model_path:
            self.tokenizer.pad_token_id = 151643
            self.tokenizer.eos_token_id = 151643
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '</s>'

        # 停止条件定义
        self.target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
        self.stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(self.target_sequences, self.tokenizer)])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.curr_eos = [151645, 151643]
        self.search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    def _extract_query(self, text):
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[-1] if matches else None

    def _search(self, query):
        """调用本地检索服务"""
        if not query or not query.strip():
            print("[WARNING] Empty query passed to search function.")
            return ""

        payload = {"queries": [query], "topk": 3, "return_scores": True}
        try:
            response = requests.post(
                self.retrieve_path,
                json=payload,
                proxies={"http": None, "https": None},
                timeout=10
            )
            response.raise_for_status()
            json_data = response.json()
            results = json_data.get("result", [])
        except requests.exceptions.Timeout:
            print("[ERROR] Search request timed out.")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed: {e}")
            return ""
        except ValueError as e:
            print(f"[ERROR] Failed to decode JSON: {e}")
            return ""

        if not results:
            print("[INFO] No results returned from search.")
            return ""

        def _format_results(retrieval_result):
            text = ''
            for idx, doc in enumerate(retrieval_result):
                content = doc['document']['contents']
                title = content.split("\n")[0]
                body = "\n".join(content.split("\n")[1:])
                text += f"Doc {idx+1}(Title: {title}) {body}\n"
            return text

        return _format_results(results[0])

    def gen(self, question, history=None):
        """执行完整的思考-检索-再思考-回答流程"""
        if history is None:
            history = []

        question = question.strip()
        # if question[-1] != '?':
        #     question += '?'

        system_prompt = (
            "根据要求，回答问题。你必须遵守思考-检索-思考-回答的推理模式。\n"
            "思考：对问题进行推理，尝试解答。推理过程中，如果你发现涉及某些法律条文，则进入检索步骤。\n"
            "检索：请把需要检索的关键词放在 <search> 和 </search> 标签之间，调用搜索引擎。例如：<search> 民法典 盗窃罪 </search>。\n"
            "系统将返回最相关的搜索结果，并置于 <information> 和 </information> 标签之间。根据返回的结果，继续下一步思考。\n"
            "再次思考：基于检索结果，继续对问题进行推理。如果没有帮助，则修改关键词重新检索；如果有把握得到最终答案，则进入回答。\n"
            "回答：在 <answer> 和 </answer> 标签内提供最终答案。\n"
            f"以下是需要回答的问题：{question}\n"
        )

        # 构造prompt
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": system_prompt}],
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            prompt = system_prompt

        cnt = 0
        print('\n\n################# [Start Reasoning + Searching] ##################\n\n')

        while True:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=2500,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.3
            )

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            if outputs[0][-1].item() in self.curr_eos or cnt > 5:
                response = output_text
                break

            tmp_query = self._extract_query(output_text)
            if tmp_query:
                print(f'[debug] search query="{tmp_query}"')
                search_results = self._search(tmp_query)
            else:
                search_results = ""

            search_text = self.search_template.format(output_text=output_text, search_results=search_results)
            prompt += search_text
            cnt += 1

        # 更新历史记录
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response})

        return response, history

if __name__ == "__main__":
    llm = LLM_retriever(
        model_path="/F00120250029/panghuaiwen/legal_LLM/model/Qwen/Qwen3-8B",
        retrieve_path="http://127.0.0.1:8006/retrieve"
    )
    resp, hist = llm.gen("经依法审查查明：2023年2月16日22时许，被告人许宏坤饮酒后驾驶**牌小型普通客车（车牌号：京M****3）行驶至本市房山区京苏路城铁桥下时遇民警设卡检查改变线路，后民警追赶至大件路阎村东地铁站时将其查获。经检测，被告人许宏坤血液中酒精含量为188.6mg/100ml，属于醉酒后驾驶机动车。被告人许宏坤于2023年2月16日被民警查获归案，后如实供述了上述事实。认定上述事实的证据如下：1.被告人的供述与辩解：被告人许宏坤的供述；2.鉴定意见：检验报告；3.书证：呼气酒精检测单、当事人血样提取登记表、驾驶人信息查询结果单、机动车信息查询结果单；4.视听资料、电子数据：查获、呼气及抽血检验视频；5.其它证明材料：接处警单、受案登记表、立案决定书、查获经过、到案经过、身份证明、公安交通管理行政强制措施凭证。上述证据收集程序合法，内容客观真实，足以认定指控事实。被告人许宏坤对指控的犯罪事实和证据没有异议，并自愿认罪认罚。给出指控被告人的罪名与相关法条。")
    print("Final Answer:\n", resp)
