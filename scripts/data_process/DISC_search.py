"""
Generate parquet from local JSONL dataset with NQ-style schema
Fully compatible with RLHFDataset
"""
import os
import json
import argparse
from datasets import Dataset

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def make_prefix(dp):
    """Chinese prompt模板"""
    q = dp["input"].strip()
    # template = (
    #     "根据要求，回答问题。\n"
    #     "每次获得新信息后，你必须首先在 <think> 和 </think> 标签内进行推理。\n"
    #     "推理完成后，如果你发现自己缺少某些知识，可以通过 <search> 【查询词】 </search> 调用搜索引擎，"
    #     "系统将返回最相关的搜索结果，并置于 <information> 和 </information> 标签之间。\n"
    #     "你可以根据需要进行多次搜索。\n"
    #     "如果你认为无需进一步获取外部知识，即可直接在 <answer> 和 </answer> 标签内提供答案，无需详细说明。"
    #     "例如：<answer> 北京 </answer>。\n"
    #     f"问题：{q}\n"
    # )
    template = (
        "根据要求，回答问题。你必须遵守思考-检索-思考-回答的推理模式。\n"
        "思考：对问题进行推理，尝试解答。推理过程中，如果你发现涉及某些法律条文，则进入检索步骤。\n"
        "检索：请把需要检索的关键词放在 <search> 和 </search> 标签之间，调用搜索引擎。例如：<search> 民法典 盗窃罪 </search>。系统将返回最相关的搜索结果，并置于 <information> 和 </information> 标签之间。根据返回的结果，继续下一步思考。"
        "再次思考：基于检索结果，继续对问题进行推理。搜索结果是否有助于解决问题？如果没有帮助，则修改关键词，回到上一步检索;如果结合先前的思考和检索结果，你已经有把握得到最终答案，则进入下一步回答。\n"
        "回答：直接在 <answer> 和 </answer> 标签内提供最终答案，无需详细说明。例如：<answer> 北京 </answer>。\n"
        f"以下是需要回答的问题：{q}\n"
    )
    return template

def normalize_answer_field(ex):
    """确保 output 为 list[str]"""
    ans = ex.get("output", [])
    if isinstance(ans, str):
        return [ans]
    if isinstance(ans, (list, tuple)):
        return [str(a) for a in ans]
    return [str(ans)]

def build_dataset(examples, split, data_source):
    data = []
    for idx, ex in enumerate(examples):
        record = {
            "id": str(ex.get("id", f"{split}_{idx}")),
            "question": str(ex["input"]),
            "golden_answers": normalize_answer_field(ex),
            "data_source": data_source,
            "prompt": [
                {"role": "user", "content": make_prefix(ex)}
            ],
            "ability": "fact-reasoning",
            "reward_model": {
                "ground_truth": {"target": normalize_answer_field(ex)},
                "style": "rule"
            },
            "extra_info": {"question":str(ex["input"]),"index": idx, "split": split},
        }
        data.append(record)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Input JSONL path")
    parser.add_argument("--output_dir", default="./output", help="Directory to save parquet")
    parser.add_argument("--data_name", default="test")
    parser.add_argument("--data_source", default="DISC")
    parser.add_argument("--start", default=0,type=int)
    parser.add_argument("--end", default=None,type=int)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    examples = load_jsonl(args.data_path)
    assert args.start <= args.end
    examples=examples[args.start:args.end]


    records = build_dataset(examples, split=args.data_name, data_source=args.data_source)

    # 确保Arrow推断的类型与原始一致
    ds = Dataset.from_list(records)
    out_path = os.path.join(args.output_dir, f"{args.data_name}.parquet")
    ds.to_parquet(out_path)

    print(f"Saved {len(ds)} rows to {out_path} with NQ-style schema")
