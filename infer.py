import transformers
import torch
import random
from datasets import load_dataset
import requests

question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"

# Model ID and device setup
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
model_id = "/linzhihang/huaiwenpang/legal_LLM/Search-R1/Search-R1/verl_checkpoints/nq-search-r1-ppo-qwen2.5-7b-em/actor/global_step_100"

# model_id = "/linzhihang/huaiwenpang/legal_LLM/Search-R1/Search-R1/verl_checkpoints/nq-search-r1-ppo-llama3.2-3b-em/actor/global_step_1000"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

question = question.strip()
if question[-1] != '?':
    question += '?'
curr_eos = [151645, 151643] # for Qwen2.5 series models

curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# Prepare the message
prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

#use tokenizer's eos and pad token
# eos_token_id=tokenizer.eos_token_id
# pad_token_id=tokenizer.pad_token_id
# curr_eos=[eos_token_id]
# print(f'[debug] eos_token_id={eos_token_id},pad_token_id={pad_token_id}')

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

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
    if not query or not query.strip():
        print("[WARNING] Empty query passed to search function.")
        return ""
    
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    # results = requests.post("http://127.0.0.1:8006/retrieve", json=payload).json()['result']
    try:
        response = requests.post("http://127.0.0.1:8006/retrieve", 
                                json=payload,
                                proxies={"http": None, "https": None},  # 禁用所有代理 
                                timeout=10)
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
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


# Initialize the stopping criteria
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

cnt = 0

if tokenizer.chat_template:
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
print(f'**[prompt]:{prompt}')
# Encode the chat-formatted prompt and move it to the correct device
while True:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Generate text with the stopping criteria
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,#for qwen2.5
        # pad_token_id=pad_token_id,
        do_sample=True,
        temperature=0.3
    )

    if outputs[0][-1].item() in curr_eos:
        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f'**after search time {cnt}')
        print(f"**final result:\n{output_text}")
        break

    generated_tokens = outputs[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # print(f'[debug] output "{output_text}"...')
    tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
    if tmp_query:
        
        print(f'**[debug]start searching\n "{tmp_query}"...')
        search_results = search(tmp_query)
        print(f'**[debug]searching result :\n"{search_results}"')
    else:
        search_results = ''

    search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
    prompt += search_text
    cnt += 1
    # print(f'[search_text in NO.{cnt}]:{search_text}')
