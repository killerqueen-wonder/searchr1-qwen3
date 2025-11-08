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
    q = dp["question"].strip()
    template = (
        "根据要求，回答问题。\n"
        "每次获得新信息后，你必须首先在 <think> 和 </think> 标签内进行推理。\n"
        "推理完成后，如果你发现自己缺少某些知识，可以通过 <search> 【查询词】 </search> 调用搜索引擎，"
        "系统将返回最相关的搜索结果，并置于 <information> 和 </information> 标签之间。\n"
        "你可以根据需要进行多次搜索。\n"
        "如果你认为无需进一步获取外部知识，即可直接在 <answer> 和 </answer> 标签内提供答案，无需详细说明。"
        "例如：<answer> 北京 </answer>。\n"
        f"问题：{q}\n"
    )
    return template

def normalize_answer_field(ex):
    """确保 golden_answers 为 list[str]"""
    ans = ex.get("golden_answers", [])
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
            "question": str(ex["question"]),
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
            "extra_info": {"index": idx, "split": split},
        }
        data.append(record)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Input JSONL path")
    parser.add_argument("--output_dir", default="./output", help="Directory to save parquet")
    parser.add_argument("--data_name", default="test")
    parser.add_argument("--data_source", default="legal_exam")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    examples = load_jsonl(args.data_path)
    records = build_dataset(examples, split=args.data_name, data_source=args.data_source)

    # 确保Arrow推断的类型与原始一致
    ds = Dataset.from_list(records)
    out_path = os.path.join(args.output_dir, f"{args.data_name}.parquet")
    ds.to_parquet(out_path)

    print(f"Saved {len(ds)} rows to {out_path} with NQ-style schema")
