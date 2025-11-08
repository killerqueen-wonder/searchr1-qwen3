import sys, os
import argparse
sys.path.append('/caizhenyang/panghuaiwen/legal_LLM/searchr1-qwen3')

from verl.model_merger.base_model_merger import parse_args, generate_config_from_args
from verl.model_merger.fsdp_model_merger import FSDPModelMerger

if __name__ == "__main__":
    # 模拟命令: python run_merger.py merge --backend fsdp --local_dir ... --target_dir ...
    args = parse_args()
    config = generate_config_from_args(args)
    merger = FSDPModelMerger(config)
    merger.merge_and_save()
