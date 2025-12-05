import json
import os
import warnings
from typing import List, Dict, Optional, Any,Tuple
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import sys
import os
from text2vec import SentenceModel, semantic_search
import torch
import time
import json

from rank_bm25 import BM25Okapi
import jieba
import lawa

import re

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        torch.cuda.empty_cache()

        return query_emb

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


class BM25WeightRetriever(BaseRetriever):#rank bm25+jieba or lawa
    def __init__(self, config):
        super().__init__(config)
        print(f'[debug][BM25Retriever]weight factor={config.bm25_weight_factor}')

        
        #自定义词典
        if len(config.dictionary_path) !=0:
            print(f'[debug] load userdict :{config.dictionary_path}')
            lawa.load_userdict(config.dictionary_path)

        # 加载语料库（必须是 jsonl，每条含 content 字段）
        self.corpus = load_corpus(self.corpus_path)

        # 对语料库进行分词与预处理
        # self.docs_raw = [doc["content"] for doc in self.corpus]

        #对法律名和编号加权
        enhanced_docs = []
        for doc in self.corpus:
            text = doc["content"]

            # 提取 “法律名”：从第一次出现“中华人民共和国”末尾，到后第一次出现空格
            law_name = ""
            m1 = re.search(r"(?<=中华人民共和国)\s*(.*?)(?=\s|$)", text, flags=re.S)
            if m1:
                law_name = m1.group(1).strip()

            # 提取 “法条编号”：从第一次出现“\n  第”，到之后第一次出现“条\n”
            article_id = ""
            m2 = re.search(r"\n\s*第(.*?)条\n", text, flags=re.S)
            if m2:
                article_id = f"{m2.group(1)}"

            # 权重增强：重复添加若干次（可调）
            if config.bm25_weight_factor:
                weight_factor=config.bm25_weight_factor
                # print(f'[debug]weight factor={weight_factor}')
            else:
                weight_factor = 3   # 可调
                # print(f'[debug]default weight factor={weight_factor}')
            weighted_tokens = []
            if law_name:
                weighted_tokens.extend([law_name] * weight_factor)
            if article_id:
                weighted_tokens.extend([article_id] * weight_factor)

            enhanced_docs.append({
                "content": text,
                "weighted": " ".join(weighted_tokens)
            })

        # 替换为增强后的内容
        self.docs_raw = [
            (doc["weighted"] + " " + doc["content"])
            for doc in enhanced_docs
        ]
        self.docs_tokenized = [list(lawa.cut(text)) for text in self.docs_raw]
        
        # 构建 BM25
        self.bm25 = BM25Okapi(self.docs_tokenized,k1=1.5, b=0.5)

        
        self.max_process_num = 8

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk

        
        def clean_string_regex(text):
            # 使用正则表达式移除标点符号
            pattern = r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?/\'"、。，！？；：「」『』（）【】《》﹁﹂﹃﹄‘’“”～﹏丶]'
            return re.sub(pattern, ' ', text)
           
        #清洗检索词中的符号
        query=clean_string_regex(query)

        #转换数字
        def num_to_chinese(num):
            if not (0 <= num <= 9999): # 限制在0-9999范围内
                return num
            num_map = {
                '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
                '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
            }
            units = ['', '十', '百', '千'] # 单位只到千
            
            if num == 0:
                return '零'
                
            digits = list(str(num))
            digits.reverse() # 反转以便从个位开始处理
            
            result_parts = []
            zero_flag = False # 标记是否需要加零
            
            for i, digit in enumerate(digits):
                current_unit = units[i]
                if digit == '0':
                    zero_flag = True # 标记中间有0
                else:
                    if zero_flag and result_parts:
                        # 如果之前有0且结果不为空，则加一个零
                        result_parts.append('零')
                    result_parts.append(num_map[digit] + current_unit)
                    zero_flag = False # 重置零标记
                    
            # 反转结果列表并拼接
            result_parts.reverse()
            res_str = ''.join(result_parts)
            
            # 特殊处理：10-19的数字，如10应为"十"而非"一十"
            if 10 <= num <= 19:
                res_str = res_str.replace('一十', '十')
                
            return res_str

        def replace_numbers_with_chinese(text):
            """
            将字符串中的所有连续数字替换为中文数字
            """
            def replace_match(match):
                num_str = match.group()
                # 将字符串转换为整数，然后调用你的num_to_chinese函数
                return num_to_chinese(int(num_str))
            
            # 使用re.sub进行替换
            return re.sub(r'\d+', replace_match, text)

        query = replace_numbers_with_chinese(query)

        # query_tokens = list(jieba.cut_for_search(query))
        # query_tokens = list(lawa.cut_for_search(query))
        query_tokens = list(lawa.cut(query))

        print("[DEBUG] Query Tokens:", query_tokens)

        scores = self.bm25.get_scores(query_tokens)

        

        if len(scores) == 0:
            if return_score:
                return [], []
            return []

        # 取 topk
        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )[:num]

        # print("\n[DEBUG] Top-k Documents Info:")
        # for rank, (doc_id, score) in enumerate(ranked):
        #     print(f"\n[DEBUG] Rank {rank+1}: Doc {doc_id}, Score = {score}")
        #     print("[DEBUG] Doc Tokens:", self.docs_tokenized[doc_id])

        doc_ids = [idx for idx, _ in ranked]
        top_scores = [score for _, score in ranked]

        results = load_docs(self.corpus, doc_ids)

        if return_score:
            return results, top_scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results

class BM25Retriever(BaseRetriever):#rank bm25+jieba or lawa
    def __init__(self, config):
        super().__init__(config)
        

        
        #自定义词典
        if len(config.dictionary_path) !=0:
            print(f'[debug] load userdict :{config.dictionary_path}')
            lawa.load_userdict(config.dictionary_path)

        # 加载语料库（必须是 jsonl，每条含 content 字段）
        self.corpus = load_corpus(self.corpus_path)

        # 对语料库进行分词与预处理
        self.docs_raw = [doc["content"] for doc in self.corpus]

        
        self.docs_tokenized = [list(lawa.cut(text)) for text in self.docs_raw]
        
        # 构建 BM25
        self.bm25 = BM25Okapi(self.docs_tokenized,k1=1.5, b=0.5)

        
        self.max_process_num = 8

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk

        
        def clean_string_regex(text):
            # 使用正则表达式移除标点符号
            pattern = r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?/\'"、。，！？；：「」『』（）【】《》﹁﹂﹃﹄‘’“”～﹏丶]'
            return re.sub(pattern, ' ', text)
           
        #清洗检索词中的符号
        query=clean_string_regex(query)

        #转换数字
        def num_to_chinese(num):
            if not (0 <= num <= 9999): # 限制在0-9999范围内
                return num
            num_map = {
                '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
                '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
            }
            units = ['', '十', '百', '千'] # 单位只到千
            
            if num == 0:
                return '零'
                
            digits = list(str(num))
            digits.reverse() # 反转以便从个位开始处理
            
            result_parts = []
            zero_flag = False # 标记是否需要加零
            
            for i, digit in enumerate(digits):
                current_unit = units[i]
                if digit == '0':
                    zero_flag = True # 标记中间有0
                else:
                    if zero_flag and result_parts:
                        # 如果之前有0且结果不为空，则加一个零
                        result_parts.append('零')
                    result_parts.append(num_map[digit] + current_unit)
                    zero_flag = False # 重置零标记
                    
            # 反转结果列表并拼接
            result_parts.reverse()
            res_str = ''.join(result_parts)
            
            # 特殊处理：10-19的数字，如10应为"十"而非"一十"
            if 10 <= num <= 19:
                res_str = res_str.replace('一十', '十')
                
            return res_str

        def replace_numbers_with_chinese(text):
            """
            将字符串中的所有连续数字替换为中文数字
            """
            def replace_match(match):
                num_str = match.group()
                # 将字符串转换为整数，然后调用你的num_to_chinese函数
                return num_to_chinese(int(num_str))
            
            # 使用re.sub进行替换
            return re.sub(r'\d+', replace_match, text)

        query = replace_numbers_with_chinese(query)

        # query_tokens = list(jieba.cut_for_search(query))
        # query_tokens = list(lawa.cut_for_search(query))
        query_tokens = list(lawa.cut(query))

        print("[DEBUG] Query Tokens:", query_tokens)

        scores = self.bm25.get_scores(query_tokens)

        

        if len(scores) == 0:
            if return_score:
                return [], []
            return []

        # 取 topk
        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )[:num]

        # print("\n[DEBUG] Top-k Documents Info:")
        # for rank, (doc_id, score) in enumerate(ranked):
        #     print(f"\n[DEBUG] Rank {rank+1}: Doc {doc_id}, Score = {score}")
        #     print("[DEBUG] Doc Tokens:", self.docs_tokenized[doc_id])

        doc_ids = [idx for idx, _ in ranked]
        top_scores = [score for _, score in ranked]

        results = load_docs(self.corpus, doc_ids)

        if return_score:
            return results, top_scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json',
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            '''
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            '''

            # 创建 GPU 资源管理器并设置每个 GPU 的显存限制
            res_list = []
            #允许列表形式的内存限制
            if len(config.gpu_memory_limit_per_gpu)==1:  
                config.gpu_memory_limit_per_gpu = config.gpu_memory_limit_per_gpu * len(config.gpu_ids)  
            for gpu_id,mem_lim in zip(config.gpu_ids,config.gpu_memory_limit_per_gpu):
                res = faiss.StandardGpuResources()
                res.setTempMemory(mem_lim * 1024 * 1024 * 1024)  # 单位：字节
                res.noTempMemory()  # 禁用临时内存分配
                res_list.append(res)

            # 启用显存优化策略（分片、混合精度）
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True      # 使用混合精度（降低显存占用）
            co.shard = True            # 分片到多个 GPU
            co.copyInvertedListsOnGpu = True  # 将倒排列表复制到 GPU
            print(f'使用GPU{config.gpu_ids},内存限制为{config.gpu_memory_limit_per_gpu}')
            # 将索引迁移到多个 GPU
            # self.index = faiss.index_cpu_to_all_gpus(self.index, co=co, gpus=config.gpu_ids)
            self.index = faiss.index_cpu_to_gpu_multiple_py(res_list, self.index, co=co, gpus=config.gpu_ids)
            
        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name = self.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            results.extend(batch_results)
            scores.extend(batch_scores)
            
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()
            
        if return_score:
            return results, scores
        else:
            return results

class Text2vecRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        
        # 初始化 text2vec 模型
        self.embedder = SentenceModel(config.retrieval_model_path)
        
        # 加载语料库
        self.corpus = load_corpus(self.corpus_path)
        
        # 生成对应的embedding文件名
        self.embedding_file = self._get_embedding_filename(self.corpus_path)
        
        # 加载或计算语料库嵌入
        self.corpus_embeddings = self._load_or_compute_embeddings()
        
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _get_embedding_filename(self, jsonl_filename):
        """
        根据jsonl文件名生成对应的embedding文件名
        """
        base_name = os.path.splitext(os.path.basename(jsonl_filename))[0]
        dir_name = os.path.dirname(jsonl_filename)
        embedding_filename = os.path.join(dir_name, f"{base_name}_embeddings.pt")
        return embedding_filename

    def _load_or_compute_embeddings(self):
        """
        加载预计算的嵌入向量，如果不存在则计算并保存
        """
        print(f"[INFO] 检查预计算的embedding文件: {self.embedding_file}")
        
        if os.path.exists(self.embedding_file):
            print(f"[INFO] 发现预计算的embedding文件，正在加载...")
            corpus_embeddings = torch.load(self.embedding_file)
            print(f"[INFO] 已加载 embedding shape={corpus_embeddings.shape}")
        else:
            print(f"[INFO] 未找到预计算的embedding文件，开始计算...")
            corpus_embeddings = self._compute_and_save_embeddings()
            
        return corpus_embeddings

    def _compute_and_save_embeddings(self):
        """
        计算语料库嵌入向量并保存
        """
        # 提取语料库文本内容
        corpus_texts = []
        for doc in self.corpus:
            if 'contents' in doc:
                corpus_texts.append(doc['contents'])
            elif 'text' in doc:
                corpus_texts.append(doc['text'])
            elif 'content' in doc:
                corpus_texts.append(doc['content'])
            else:
                corpus_texts.append(str(doc))
        
        print(f"[INFO] 正在计算 {len(corpus_texts)} 个文档的语义向量...")
        start_time = time.time()
        
        # 计算嵌入向量
        if torch.cuda.is_available():
            print(f"[INFO] 使用 GPU: {torch.cuda.get_device_name(0)}")
            pool = self.embedder.start_multi_process_pool()
            corpus_embeddings = self.embedder.encode_multi_process(corpus_texts, pool, normalize_embeddings=True)
            self.embedder.stop_multi_process_pool(pool)
        else:
            print("[INFO] 使用 CPU 计算")
            corpus_embeddings = self.embedder.encode(corpus_texts, normalize_embeddings=True)
        
        # 保存嵌入向量
        print(f"[INFO] 保存 embedding 到: {self.embedding_file}")
        torch.save(corpus_embeddings, self.embedding_file)
        print(f"[INFO] 已保存 shape={corpus_embeddings.shape}")
        
        end_time = time.time()
        print(f"[DEBUG] embedding 计算时间: {end_time - start_time:.2f} 秒")
        
        return corpus_embeddings

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
            
        # start_time = time.time()
        
        # 计算查询嵌入向量
        query_embedding = self.embedder.encode(query, normalize_embeddings=True)
        
        # 使用 semantic_search 进行检索
        hits = semantic_search(query_embedding, self.corpus_embeddings, top_k=num)[0]
        
        # end_time = time.time()
        # print(f"[DEBUG] t2v 单次检索时间: {end_time - start_time:.4f} 秒")
        
        # 构建结果
        results = []
        scores = []
        
        for hit in hits:
            doc_idx = hit['corpus_id']
            score = hit['score']
            
            # 获取对应文档
            doc = self.corpus[doc_idx]
            results.append(doc)
            scores.append(score)
        
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
            
        start_time = time.time()
        
        # 计算所有查询的嵌入向量
        query_embeddings = self.embedder.encode(query_list, normalize_embeddings=True)
        
        # 批量检索
        batch_results = []
        batch_scores = []
        
        for i, query_embedding in enumerate(query_embeddings):
            # 为每个查询单独进行检索（因为 semantic_search 需要单个查询向量）
            hits = semantic_search(query_embedding, self.corpus_embeddings, top_k=num)[0]
            
            results = []
            scores = []
            
            for hit in hits:
                doc_idx = hit['corpus_id']
                score = hit['score']
                
                doc = self.corpus[doc_idx]
                results.append(doc)
                scores.append(score)
            
            batch_results.append(results)
            batch_scores.append(scores)
        
        end_time = time.time()
        print(f"[DEBUG] 批量检索时间 ({len(query_list)} 个查询): {end_time - start_time:.4f} 秒")
        
        if return_score:
            return batch_results, batch_scores
        else:
            return batch_results

class HybridRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        
        # 初始化组件
        self.bm25_retriever = BM25Retriever(config)
        self.text2vec_retriever = Text2vecRetriever(config)

        self.topk = config.retrieval_topk
        self.search_depth=config.search_depth
        self.candidate_k = self.topk * self.search_depth

        
        # 融合权重
        self.w_bm25 = config.bm25_weight
        self.w_t2v = 10

    def _min_max_norm(self, scores):
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return [0.0 for _ in scores]
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk

        self.candidate_k = num * self.search_depth
        start_time = time.time()

        # bm25 检索
        bm25_results, bm25_scores = self.bm25_retriever._search(query, self.candidate_k, True)
        # print("[DEBUG][Hybrid] BM25 top candidates:")
        
        # for i, (doc, sc) in enumerate(zip(bm25_results, bm25_scores)):
        #     if i < num:
        #         print(f"  [BM25] Rank {i+1}: score={sc}, doc_content={doc['contents'] }")

        # text2vec 检索
        t2v_results, t2v_scores = self.text2vec_retriever._search(query, self.candidate_k, True)
        # print("[DEBUG][Hybrid] Text2Vec top candidates:")
        # for i, (doc, sc) in enumerate(zip(t2v_results, t2v_scores)):
        #     # print(f"  [T2V] Rank {i+1}: score={sc}, doc_id={doc['id'] if 'id' in doc else i}")
        #     if i < num:
        #         print(f"  [T2V] Rank {i+1}: score={sc}, doc_content={doc['contents'] }")
            

        # 构建候选池
        pool = {}  # doc_id -> {bm25_score, t2v_score, doc}

        def add_pool(results, scores, key):
            for doc, sc in zip(results, scores):
                doc_id = doc["id"] if "id" in doc else id(doc)
                if doc_id not in pool:
                    pool[doc_id] = {"doc": doc, "bm25": None, "t2v": None}
                pool[doc_id][key] = sc

        add_pool(bm25_results, bm25_scores, "bm25")
        add_pool(t2v_results, t2v_scores, "t2v")

        # 填补缺失得分
        for info in pool.values():
            if info["bm25"] is None:
                info["bm25"] = 0.0
            if info["t2v"] is None:
                info["t2v"] = 0.0

        # 分别取出分数供归一化
        bm25_all = [v["bm25"] for v in pool.values()]
        t2v_all = [v["t2v"] for v in pool.values()]

        bm25_norm = self._min_max_norm(bm25_all)
        t2v_norm = self._min_max_norm(t2v_all)

        # 写回归一化分数
        for (doc_id, info), bn, tn in zip(pool.items(), bm25_norm, t2v_norm):
            info["bm25_norm"] = bn
            info["t2v_norm"] = tn
            info["hybrid_score"] = self.w_bm25 * bn + self.w_t2v * tn

        # 按融合分数排序
        ranked = sorted(pool.items(), key=lambda x: x[1]["hybrid_score"], reverse=True)[:num]
        
        # print("[DEBUG][Hybrid] Final fused ranking:")
        # for rank, (doc_id, info) in enumerate(ranked, 1):
        #     print(f"  [Hybrid] Rank {rank}: score={info['hybrid_score']}, doc_id={doc_id}")

        results = [info["doc"] for _, info in ranked]
        scores = [info["hybrid_score"] for _, info in ranked]

        end_time = time.time()
        print(f"[DEBUG] hybrid单次检索时间: {end_time - start_time:.4f} 秒")
        if return_score:
            return results, scores
        return results

    def _batch_search(self, query_list, num=None, return_score=False):
        results = []
        scores = []
        for q in query_list:
            r, s = self._search(q, num, True)
            results.append(r)
            scores.append(s)
        return (results, scores) if return_score else results

class HybridFilterRetriever(HybridRetriever):
    def __init__(self, config):
        # 初始化父类 (BM25 + Text2Vec)
        super().__init__(config)
        
        # 初始化 Filter 模型 (LLM)
        print(f"[Init] Loading Filter LLM from: {config.filter_model} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.filter_model, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.filter_model, 
                device_map="auto", 
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.eval()
        except Exception as e:
            print(f"[Error] Failed to load Filter LLM: {e}")
            raise e

        # 提示词模板
        self.prompt_template = (
            "你的任务是从备选文本中评价哪些文本符合检索词。\n"
            "备选文本为：{results}\n"
            "检索词为：{query}\n"
            # "现在，从备选文本中筛选出符合检索词的文本，保留原格式，保留原文本，不要输出其他解释。"
            "现在，给出一个列表，代表你判断第几段文本符合检索词（从1开始）。例如：[1,3,4],不要输出其他解释性内容。"
            "如果全部不符合，则返回空字符串。"
        )

    def _llm_filter(self, query: str, candidates: List[Dict], scores: List[float]) -> Tuple[List[Dict], List[float]]:
        """
        使用 LLM 对候选文档进行筛选
        """
        if not candidates:
            return [], []

        # 1. 构建 Prompt 输入
        # 为了让 LLM 更好理解结构，我们将候选文档转为简化的 JSON 字符串或带序号的列表
        
        candidates_data = []
        for doc in candidates:
            # 提取关键信息给 LLM 判断，减少 token 消耗，主要是 content
            candidates_data.append({
                "content": doc.get("content", ""),
                # 可以根据需要添加其他辅助判断字段，如 law_name
            })
        
        results_str = json.dumps(candidates_data, ensure_ascii=False, indent=1)
        prompt = self.prompt_template.format(results=results_str, query=query)

        # 2. 模型推理
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=5000, # 预留足够的输出长度
                    temperature=0.1,     # 低温以保证确定性
                    do_sample=False
                )
            
            # 获取生成的文本（去掉 prompt 部分）
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            
        except Exception as e:
            print(f"[Warning] LLM Filter failed: {e}, returning original top results.")
            return candidates, scores

        # 3. 解析结果并筛选
        # 策略：如果 LLM 返回的文本中包含了文档的内容片段，则认为该文档被选中。
        # 这种方式比强制 LLM 输出严格 JSON 更鲁棒。
        
        filtered_results = []
        filtered_scores = []
        print(f"[debug]model response:{response}")

        # 如果模型返回空字符串或表示无结果
        if not response.strip():
            return [], []
        
        

        def extract_numbers_last_brackets(text):
            # 使用正则表达式查找最后一个 [...] 对
            pattern = r'\[([^\[\]]*)\]'
            matches = list(re.finditer(pattern, text))
            
            if not matches:
                return None
            
            # 获取最后一个匹配
            last_match = matches[-1]
            
            # 提取数字
            numbers = re.findall(r'\d+', last_match.group(1))
            
            return set(map(int, numbers)) if numbers else None

        #读取模型筛选的文本编号
        text_num=extract_numbers_last_brackets(response)
        print(f"[debug]model text_num:{text_num}")

        # 如果模型返回空字符串或表示无结果
        if not text_num:
            return [], []

        filtered_results = [candidates[i - 1] for i in sorted(text_num) if i - 1 < len(candidates)]
        filtered_scores = [scores[i - 1] for i in sorted(text_num) if i - 1 < len(scores)]

        # filtered_results = [item for item in candidates if (item+1) in text_num]
        # filtered_scores = [item for item in scores if (item+1) in text_num]
        
        print(f"[debug]filtered_results:{filtered_results}")
        print(f"[debug]filtered_scores:{filtered_scores}")
        return filtered_results, filtered_scores

    def _search(self, query: str, num: int = None, return_score: bool = False):
        # 1. 使用父类 HybridRetriever 进行初步检索 (Recall)
        initial_num = num if num else self.topk

        
        candidates, scores = super()._search(query, initial_num, return_score=True)
        
        start_time = time.time()
        
        # 2. 使用 LLM 进行过滤 (Precision)
        filtered_results, filtered_scores = self._llm_filter(query, candidates, scores)
        
        end_time = time.time()
        print(f"[DEBUG] LLM Filter time: {end_time - start_time:.4f} s, Input: {len(candidates)} -> Output: {len(filtered_results)}")

        if return_score:
            return filtered_results, filtered_scores
        return filtered_results
    
    
def get_retriever(config):
    
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    elif config.retrieval_method == "hybrid":
        return HybridRetriever(config)
    elif config.retrieval_method == "text2vec":
        return Text2vecRetriever(config)
    elif config.retrieval_method == "hybrid_filter":
        return HybridFilterRetriever(config)
    elif config.retrieval_method == "BM25Weight":
        return BM25WeightRetriever(config)
    else:
        return DenseRetriever(config)

#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        
        gpu_ids: List[int] = [3, 4, 5, 7],  # 新增 GPU ID 列表
        gpu_memory_limit_per_gpu =18,#新增内存限制
        port =8006,#新增port端口号
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        dictionary_path:str="",
        search_depth:int =5,
        bm25_weight:int=10,
        bm25_weight_factor:int =3,
        filter_model:str="",
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu

        self.port=port
        self.gpu_ids=gpu_ids
        self.gpu_memory_limit_per_gpu=gpu_memory_limit_per_gpu

        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.dictionary_path=dictionary_path
        self.search_depth=search_depth
        self.bm25_weight=bm25_weight
        self.bm25_weight_factor=bm25_weight_factor
        self.filter_model=filter_model
        


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    results, scores = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores
    )
    
    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--dictionary_path", type=str, default='', help="jieba dictionary for law")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--search_depth", type=int, default=5, help="hydrid search depth")
    parser.add_argument("--bm25_weight", type=int, default=10)
    parser.add_argument("--bm25_weight_factor", type=int, default=3)
    parser.add_argument("--retriever_name", type=str, default="text2vec", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument("--filter_model", type=str, default="")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')

    parser.add_argument("--port", type=int, default=8006, help="the API port")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=[3, 4, 5, 7], help="GPU device IDs to use.")
    parser.add_argument("--gpu_memory_limit_per_gpu", type=int, nargs='+', default=[18], help="GPU memory limit per GPU in GB.")


    args = parser.parse_args()
    
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method = args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,

        port=args.port,  
        gpu_ids=args.gpu_ids,  # 传递 GPU ID
        gpu_memory_limit_per_gpu=args.gpu_memory_limit_per_gpu,  # 传递显存限制

        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
        dictionary_path=args.dictionary_path,
        search_depth=args.search_depth,
        bm25_weight=args.bm25_weight,
        bm25_weight_factor=args.bm25_weight_factor,

        filter_model=args.filter_model,
    )
    

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8006
    uvicorn.run(app, host="0.0.0.0", port=config.port)
