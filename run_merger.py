# run_merger.py
import sys
import os
import argparse

# 添加模型合并器路径到Python路径
sys.path.append('/caizhenyang/panghuaiwen/legal_LLM/searchr1-qwen3')

from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.model_merger.base_model_merger import ModelMergerConfig


parser = argparse.ArgumentParser()

parser.add_argument('--local_dir', type=str)
parser.add_argument('--target_dir',type=str)
args = parser.parse_args()
# 定义配置
config = ModelMergerConfig(
    operation= "merge",
    backend= "fsdp", 
    local_dir= args.local_dir,
    target_dir= args.target_dir
    )

# 创建合并器并运行
merger = FSDPModelMerger(config)
merger.merge_and_save()