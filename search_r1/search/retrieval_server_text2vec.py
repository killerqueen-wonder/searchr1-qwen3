import json
import os
import warnings
from typing import List, Dict, Optional
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
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

class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
    
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                for hit in hits
            ]
            results = [
                {
                    'title': content.split("\n")[0].strip("\""),
                    'text': "\n".join(content.split("\n")[1:]),
                    'contents': content
                } 
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
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
            
        start_time = time.time()
        
        # 计算查询嵌入向量
        query_embedding = self.embedder.encode(query, normalize_embeddings=True)
        
        # 使用 semantic_search 进行检索
        hits = semantic_search(query_embedding, self.corpus_embeddings, top_k=num)[0]
        
        end_time = time.time()
        print(f"[DEBUG] 单次检索时间: {end_time - start_time:.4f} 秒")
        
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


def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    elif config.retrieval_method == "text2vec":
        return Text2vecRetriever(config)
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
        retrieval_batch_size: int = 128
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
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="text2vec", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
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
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8006
    uvicorn.run(app, host="0.0.0.0", port=config.port)
