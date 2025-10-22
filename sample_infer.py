
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
# prompt =question
curr_eos = [151645, 151643] # for Qwen2.5 series models

# 初始化 tokenizer 和模型
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)



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
    

# Initialize the stopping criteria
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

for i in range(3):
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
        temperature=0.7
    )
    
    # if outputs[0][-1].item() in curr_eos:
    if True:
        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(output_text)
        break