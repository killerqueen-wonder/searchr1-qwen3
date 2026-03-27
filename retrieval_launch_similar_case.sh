#!/bin/bash

# 配置路径
file_path=/mnt/nvme1n1/legal_LLM/dataset/law
corpus_file=$file_path/lecard_court_psi.jsonl
retriever_path=shibing624/text2vec-base-chinese-paraphrase

echo "=========================================="
echo "Starting Similar Case Retriever (类案检索)"
echo "Corpus: $corpus_file"
echo "=========================================="

python search_r1/search/retrieval_server_similar_case.py \
    --corpus_path $corpus_file \
    --retriever_model $retriever_path \
    --topk 6 \
    --search_depth 10 \
    --port 7007 \
    --gpu_ids 2,3 \
    "$@"