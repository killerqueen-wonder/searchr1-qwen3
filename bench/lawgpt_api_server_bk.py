import sys
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# 导入你写好的 Infer 类 (假设保存在 infer.py 中)
import sys
import json

import fire
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"

app = FastAPI()


class Infer():
    def __init__(
        self,
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "",
        prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    ):
        prompter = Prompter(prompt_template)
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        try:
            print(f"Using lora {lora_weights}")
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        except:
            print("*"*50, "\n Attention! No Lora Weights \n", "*"*50)
            
        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        
        self.base_model = base_model
        self.lora_weights = lora_weights
        self.model = model
        self.prompter = prompter
        self.tokenizer = tokenizer

    def generate_output(
        self,
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=256,
        **kwargs,
    ):
        prompt = self.prompter.generate_prompt(instruction, input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            # repetition_penalty=10.0,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        return self.prompter.get_response(output)

    def infer_from_file(self, infer_data_path):
        with open(infer_data_path) as f:
            for line in f:
                data = json.loads(line)
                instruction = data["instruction"]
                output = data["output"]
                print('=' * 100)
                print(f"Base Model: {self.base_model}    Lora Weights: {self.lora_weights}")
                print("Instruction:\n", instruction)
                model_output = self.generate_output(instruction)
                print("Model Output:\n", model_output)
                print("Ground Truth:\n", output)
                print('=' * 100)


# 初始化模型
print("Loading LawGPT Model...")
infer_engine = Infer(
    load_8bit=False,
    base_model='minlik/chinese-alpaca-plus-7b-merged',
    lora_weights='',
    prompt_template='law_template'
)
print("Model Loaded Successfully!")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/completions")
async def completions(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 1000)
    temperature = data.get("temperature", 0.1)
    top_p = data.get("top_p", 0.75)

    # 如果你的系统将输入作为 instruction，可直接传入
    try:
        output_text = infer_engine.generate_output(
            instruction=prompt, 
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens
        )
    except Exception as e:
        output_text = str(e)
        
    # 返回伪装成 OpenAI / vLLM 格式的数据
    return JSONResponse(content={
        "id": "cmpl-lawgpt",
        "object": "text_completion",
        "choices": [
            {
                "text": output_text,
                "index": 0,
                "finish_reason": "length"
            }
        ]
    })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8007)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)