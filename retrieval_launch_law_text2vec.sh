
file_path=/mnt/nvme1n1/legal_LLM/dataset/law
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/law_processed.jsonl
retriever_name=text2vec
retriever_path=shibing624/text2vec-base-chinese-paraphrase

python search_r1/search/retrieval_server_text2vec.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 6 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu \
                                            "$@"
