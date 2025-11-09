#LLM-judgement
#call api to judge the reward 
import random 
import requests
import threading
from retrying import retry
import re
from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")
gpt_url = os.getenv("OPENAI_BASE_URL")

if not api_key or not gpt_url:
    raise EnvironmentError("please set OPENAI_API_KEY or OPENAI_BASE_URL .")



evaluation_prompt="""\n扮演一个公正的评委，评估用户与两个AI助手之间的对话，以判断哪个AI助手为用户提供的服务更好。\n用户的需求是：\n{needs}\n\n\n[AI助手1对话开始]\n{dialogue1}\n[AI助手1对话结束]\n\n[AI助手2对话开始]\n{dialogue2}\n[AI助手2对话结束]\n\n你的评估需要考虑AI助手回复是否满足以下标准：1.准确，专业。该回答引用条例正确且有效，引用法条来源明确，符合客观事实。
2.匹配问题，完整作答。该回答针对问题作答，没有遗漏关键点，没有偏题离题。多大程度上考虑到问题的主要方面，以及可能的额外情况，前提条件和后续步骤，潜在的风险。
3.逻辑清晰，推理合理。该回答根据前提依据，合理推导结论，推理过程清晰有层次（例如“事实-法律-结论”）。
4.实用。该回答提供可行的解决方案和路径。\n\n请您首先针对评估参考中的每一项规定进行细致的分析，分别评估两个AI助手在对话中是否正确地遵循了这些规定。\n\n随后，根据分析结果比较两个AI助手的表现。您必须仅基于AI助手回答的与评估参考的一致性来做出判断，而忽略回答的详细性和全面性。\n\n最后，严格按照以下格式输出您的最终结论：如果AI助手 1 表现更好，则输出“[[1]]”；如果AI助手 2 表现更好，则“[[2]]”；如果平局，则输出“[[3]]”。\n"""

## 定义调用 gpt4 的函数 
@retry(wait_fixed=2000, stop_max_attempt_number=10)
def call_api_timelimit(messages,api_key,gpt_url):
    class InterruptableThread(threading.Thread):
        def __init__(self,messages,api_key,gpt_url):
            threading.Thread.__init__(self)
            self.result = None
            self.messages = messages
            self.api_key = api_key
            self.gpt_url = gpt_url
            self.model_name='gpt-4o-2024-11-20'#裁判模型

        def run(self):
            try:
                
                client = OpenAI(api_key=self.api_key, base_url=self.gpt_url)

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    
                    temperature=0.2,
                    stream=False
                )
                response_text = response.choices[0].message.content.strip()
                self.result = response_text
            except Exception as e:
                print(e)
    it = InterruptableThread(messages,api_key,gpt_url)
    it.start()
    # 时间
    timeout_duration = 200
    it.join(timeout_duration)
    if it.is_alive() or it.result is None:
        print('时间进程出错')
        raise Exception("API调用超时")
    else:
        return it.result

def response(message,api_key,gpt_url):
    messages = []
    messages.append({"role": "user", "content": message})
    response_text = call_api_timelimit(messages,api_key,gpt_url)
    return response_text

## score 中 ，第一个分数是 ground_truth ， 第二个分数是 model 
def call_evaluate(ground_truth,model_dialogue,needs,evaluation_prompt,api_key,gpt_url):

    if random.random() < 0.5:
        input = evaluation_prompt.format(dialogue1=ground_truth,dialogue2=model_dialogue,needs=needs)
        input = input.strip()
        evaluation_result = response(input,api_key,gpt_url)
        score = judge_score(evaluation_result)
    else:
        input = evaluation_prompt.format(dialogue1=model_dialogue,dialogue2=ground_truth,needs=needs)
        input = input.strip()
        evaluation_result = response(input,api_key,gpt_url)
        score = judge_score(evaluation_result)
        score[0],score[1] = score[1] , score[0]
    return input , evaluation_result ,  score

def judge_score(evaluation_result):
        review = evaluation_result
        try:
            label_content = review.strip()
            label = re.findall(r"\[\[(\d)\]\]", label_content)
            if label:
                label = label[-1]
                if label == "1" :
                    return [10,0]
                elif label == "2" :
                    return [0,10]
                elif label == "3" :
                    return [5,5]
                else:
                    return [-1,-1]
            else:
                return [-1,-1]
        except Exception as e:
            print(e)
            print('error', review)
            return [-1,-1]
        
def compute_score(solution_str, ground_truth,extra_info, format_score=0.0, score=1.0):
    """The scoring function for LLM judgement.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        
        format_score: the score for the format
        score: the score for the correct answer

    """
    needs=extra_info['question']
    if len(needs) ==0:
        print('[debug] cant read questions' )
    input,evaluation_result,compare_score=call_evaluate(ground_truth,solution_str,needs,evaluation_prompt,api_key,gpt_url)
    # print(f'[debug]input:{input}')
    # print(f'[debug]evaluation_result:{evaluation_result}')
    if compare_score:
        if compare_score == [10,0] :#ground truth is better
            return format_score
        elif compare_score == [0,10] :#model is better
            return score
        elif compare_score == [5,5] :#tie
            return score
        elif compare_score == [-1,-1] :#bad case
            return 0
        
    return 0
    
