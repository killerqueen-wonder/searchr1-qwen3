
import os
import json
import torch
import logging
import argparse
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
import transformers
from transformers import set_seed
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel 
import re
import requests
from transformers import TextStreamer
import torch


#从本地模型输出各评测集的回答，支持retriever
os.umask(0)
logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

def list_to_dict(data_list):
    classified_dict = {}
    for item in data_list:
        task_name = item["task_name"]
        if task_name not in classified_dict:
            classified_dict[task_name] = [item]
        else:
            classified_dict[task_name].append(item)
    return classified_dict


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.query_prompt = ""
        raw_data = {}
        ### dataset 已经 load json 文件了
        with open(data_path) as f:
            raw_data = json.load(f)
        self.dataset = [item for sublist in raw_data.values() for item in sublist]  

    def __getitem__(self ,  index):
        # 在这里实现获取单个样本的逻辑
        sample = self.dataset[index]
        return sample

    def __len__(self):
        return len(self.dataset)
    
    def collate_fn(self, batch):
        return batch

class LLM:
    def __init__(self,model_path):
        self.model_path = model_path
        print("--------------加载模型路径为：---------------\n",model_path)
        ## 定义 model 变量 
        if "huatuo" in args.model_path.lower():
            model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
            model.generation_config = GenerationConfig.from_pretrained(args.model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        if "Qwen-72B-Chat" not in args.model_path and "deepseek" not in args.model_path and "DISC-LawLLM" not in args.model_path:
            model.cuda().eval()
        

        if 'PMC_LLaMA_13B' in args.model_path or 'zhongjing' in args.model_path:
            left_tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_path, padding_side='left')
        else:
            left_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')

        if 'Qwen' in args.model_path:
            left_tokenizer.pad_token_id = 151643
            left_tokenizer.eos_token_id = 151643
        if  left_tokenizer.pad_token is None:
            # left_tokenizer.pad_token = '<PAD>'
            left_tokenizer.pad_token = '</s>'
        
        self.model = model
        self.left_tokenizer = left_tokenizer
    
    def gen(self, query , history = [], model_prompt=""):
        
        
        if "qwen" in self.model_path.lower():
            if history == [] :
                history.append({"role":"system","content":model_prompt})
            history.append({"role": "user", "content": query})

            text = self.left_tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.left_tokenizer(text, return_tensors="pt")
            inputs = inputs.to('cuda')
            response_ids = self.model.generate(**inputs, max_new_tokens=8000)[0][len(inputs.input_ids[0]):].tolist()
            response = self.left_tokenizer.decode(response_ids, skip_special_tokens=False)
            # response = self.left_tokenizer.decode(response_ids, skip_special_tokens=True)

            # Update history
            history.append({"role": "assistant", "content": response})
            return response, history
    


# Define the custom stopping criterion
# class StopOnSequence(transformers.StoppingCriteria):
#     def __init__(self, target_sequences, tokenizer):
#         # Encode the string so we have the exact token-IDs pattern
#         self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
#         self.target_lengths = [len(target_id) for target_id in self.target_ids]
#         self._tokenizer = tokenizer

#     def __call__(self, input_ids, scores, **kwargs):
#         # Make sure the target IDs are on the same device
#         targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

#         if input_ids.shape[1] < min(self.target_lengths):
#             return False

#         # Compare the tail of input_ids with our target_ids
#         for i, target in enumerate(targets):
#             if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
#                 return True

#         return False

import torch.nn.functional as F

@torch.no_grad()
def stream_until_search(model, tokenizer, input_ids, max_new_tokens=512):
    streamer = SearchStopStreamer(tokenizer)
    
    generated = input_ids
    past_key_values = None

    for _ in range(max_new_tokens):
        outputs = model(
            generated if past_key_values is None else generated[:, -1:],
            use_cache=True,
            past_key_values=past_key_values
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # sampling
        probs = F.softmax(logits / 0.3, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # decode **only this token** and send to streamer
        text = tokenizer.decode(next_token[0])
        streamer.on_finalized_text(text)

        # append token
        generated = torch.cat((generated, next_token), dim=1)

        # 检查是否应该停止
        if streamer.done:
            break

    return streamer.search_content, generated


# class SearchStopStreamer(TextStreamer):
#     def __init__(self, tokenizer, stop_sequences):
#         super().__init__(tokenizer)
#         self.stop_sequences = stop_sequences
#         self.buffer = ""

#     def put(self, value):
#         # decode newly generated ids
#         text = self.tokenizer.decode(value, skip_special_tokens=False)
#         self.buffer += text

#         # detect any target sequence in the buffer (NOT only at the end!)
#         for seq in self.stop_sequences:
#             if seq in self.buffer:
#                 raise GenerationStop

#         # normal streaming behavior (optional)
#         return super().put(value)



class SearchStopStreamer(TextStreamer):
    """
    流式监控模型输出。如果检测到 </search> 则触发提前停止。
    """
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.buffer = ""          # 累积的全部流式文本
        self.done = False         # 控制 generate 终止
        self.search_content = ""  # 保存 <search> ... </search> 内容
    
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """
        每次模型输出新文本时，都会回调这里。
        """
        if self.done:
            return

        self.buffer += text

        # 检查是否出现完整的 </search>
        if "</search>" in self.buffer:
            # 从 buffer 中提取 <search>...</search>
            m = re.search(r"<search>(.*?)</search>", self.buffer, flags=re.DOTALL)
            if m:
                self.search_content = m.group(1).strip()

            self.done = True  # 通知外部应该停止 generate
            
        # 打印到屏幕仅为调试，可保留可删除
        print(text, end="", flush=True)



class LLM_retriever:
    """
    实现一个“思考-检索-再思考-回答”的闭环生成系统。
    """
    def __init__(self, model_path, api_key=None, api_url=None, max_turn=5,retrieve_path="http://127.0.0.1:8006/retrieve"):
        self.model_path = model_path
        
        self.api_key = api_key
        self.api_url = api_url
        self.retrieve_path = retrieve_path
        self.max_turn=max_turn

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
        # self.stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(self.target_sequences, self.tokenizer)])
        self.streamer = SearchStopStreamer(self.tokenizer)


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

    def gen(self, query ,
            #  history = [], model_prompt=""
            ):
        """执行完整的思考-检索-再思考-回答流程"""
        

        question = query.strip()
        # if question[-1] != '?':
        #     question += '?'

        system_prompt = (
            "根据要求，回答问题。你必须遵守思考-检索-思考-回答的推理模式。\n"
            "思考：对问题进行推理，尝试解答。推理过程中，如果你发现涉及某些法律条文，则进入检索步骤。\n"
            "检索：请把需要检索的关键词放在 <search> 和 </search> 标签之间，调用搜索引擎。例如：<search> 民法典 盗窃罪 </search>。\n"
            "系统将返回最相关的搜索结果，并置于 <information> 和 </information> 标签之间。根据返回的结果，继续下一步思考。\n"
            "再次思考：基于检索结果，继续对问题进行推理。如果没有帮助，则修改关键词重新检索；如果有把握得到最终答案，则进入回答。\n"
            "回答：在 <answer> 和 </answer> 标签内提供最终答案。例如：<answer> 北京 </answer>\n"
            f"以下是需要回答的问题：{question}\n"
        )

        
        # if history == [] :
        #     history.append({"role":"system","content":model_prompt})
        # history.append({"role": "user", "content": system_prompt})

        # if self.tokenizer.chat_template:
        #     prompt = self.tokenizer.apply_chat_template(
        #         history,
        #         tokenize=False,
        #         add_generation_prompt=True
        #             )
            
        # else:
            # prompt = system_prompt
        
        prompt = system_prompt
        cnt = 0
        print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
        print(f'**[prompt]:{prompt}')

        while True:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            
            # outputs = self.model.generate(
            #     input_ids,
            #     max_new_tokens=1500,
            #     # stopping_criteria=self.stopping_criteria,
            #     streamer=self.streamer,
            #     pad_token_id=self.tokenizer.eos_token_id,
            #     do_sample=True,
            #     temperature=0.3,
            #     repetition_penalty	=1.1
            # )
            search_content, generated_ids = stream_until_search(
                                            self.model, 
                                            self.tokenizer,
                                            input_ids,
                                            max_new_tokens=1500,
                                            # temperature=0.3,
                                            # repetition_penalty	=1.1,
                                            # pad_token_id=self.tokenizer.eos_token_id,
                                            )

            

            instruct=''
            # generated_tokens = outputs[0][input_ids.shape[1]:]
            # output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f'[debug] output_text="{output_text}"')

            if generated_ids[0][-1].item() in self.curr_eos or cnt > self.max_turn:
            # if outputs[0][-1].item() in self.curr_eos or cnt > self.max_turn:
                response = output_text
                print(f'[debug]search turn:{cnt}')
                print(f'[debug]final answer:{response}')
                break

            tmp_query = self._extract_query(output_text)
            print(f'[debug] search query="{tmp_query}"')
            if tmp_query and( cnt < self.max_turn):
                
                search_results = self._search(tmp_query)
            
            elif cnt==self.max_turn:
                search_results=''
                instruct = "跳过检索阶段。注意：接下来必须给出最终回答！"

            else:
                search_results = ""
                instruct="检索失败。重新检索或直接回答。"

            
            print(f'**[debug]searching result :\n"{search_results}"')
            if instruct:
                print(f'**[debug]instruct :"{instruct}"')

            search_text = self.search_template.format(output_text=output_text, search_results=search_results)
            prompt += search_text
            prompt += instruct

            cnt += 1

        # # 更新历史记录
        # history.append({"role": "user", "content": question})
        # history.append({"role": "assistant", "content": response})

        return response
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')
    # Model Args
    parser.add_argument('--question', default="Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?", type=str)
    parser.add_argument('--model_path', default="/caizhenyang/panghuaiwen/legal_LLM/RL_ckp/legal_exam-search-r1-ppo-qwen3-8b-em/global_step_100/actor", type=str)
    parser.add_argument('--retrieve_path', default="http://127.0.0.1:8006/retrieve", type=str)
    parser.add_argument('--retriever', default=False, type=bool)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_turn', default=5, type=int)
    
    args = parser.parse_args()

    set_seed(args.seed)
    model_path=args.model_path

    if args.retriever:
        llm=LLM_retriever(model_path,retrieve_path=args.retrieve_path,max_turn=args.max_turn)
    else:
        llm= LLM(model_path)
    

    res = llm.gen(args.question)
    
    

