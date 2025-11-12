import os
import re
import torch
import requests
import transformers
from transformers.generation.utils import GenerationConfig


# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
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

        
        # Initialize the tokenizer and model
    
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")


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

        def _passages2string(retrieval_result):
            format_reference = ''
            for idx, doc_item in enumerate(retrieval_result):
                            
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            return format_reference

        return _passages2string(results[0])

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
        print(f'**[prompt]:{prompt}')

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
                print(f'**[debug]searching result :\n"{search_results}"')

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
    question='甲（民营企业销售经理）因合同诈骗罪被捕。在侦查期间，甲主动供述曾向国家工作人员乙行贿9万元，司法机关遂对乙进行追诉。后查明，甲的行为属于单位行贿，行贿数额尚未达到单位行贿罪的定罪标准。甲的主动供述构成下列哪一量刑情节？A.坦白B.立功C.自首D.准自首'
    resp, hist = llm.gen(question)
    print("Final Answer:\n", resp)
