import os
import json
import time
import re
import argparse
from typing import List, Dict, Optional, Tuple

import faiss
import torch
import numpy as np
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from text2vec import SentenceModel, semantic_search
from rank_bm25 import BM25Okapi
import lawa # 确保你安装并配置了 lawa (或使用 jieba)

# ==========================================
# 1. 基础工具与配置
# ==========================================

# def load_corpus(corpus_path: str):
#     corpus = datasets.load_dataset(
#         'json', 
#         data_files=corpus_path,
#         split="train",
#         num_proc=4
#     )
#     return corpus

def load_corpus(corpus_path: str):
    print(f"[INFO] Using native JSON loader for {corpus_path}...")
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                corpus.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARNING] 无法解析第 {i+1} 行的 JSON，已跳过。")
    print(f"[INFO] 成功加载 {len(corpus)} 条数据。")
    return corpus

def load_docs(corpus, doc_idxs):
    return [corpus[int(idx)] for idx in doc_idxs]

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# ==========================================
# 2. 专用单路检索器 (Fact & Reason)
# ==========================================

class FactText2vecRetriever:
    """仅针对 fact 字段进行向量化和检索"""
    def __init__(self, config, device="cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[INFO] FactText2vec model locked on: {self.device}")
        
        self.embedder = SentenceModel(config.retrieval_model_path, device=self.device)
        self.corpus = load_corpus(config.corpus_path)
        self.batch_size = config.retrieval_batch_size
        
        # 隔离后缀，防止覆盖旧检索系统的 embeddings
        self.embedding_file = self._get_embedding_filename(config.corpus_path)
        self.corpus_embeddings = self._load_or_compute_embeddings()

    def _get_embedding_filename(self, jsonl_filename):
        base_name = os.path.splitext(os.path.basename(jsonl_filename))[0]
        dir_name = os.path.dirname(jsonl_filename)
        return os.path.join(dir_name, f"{base_name}_fact_embeddings.pt")

    def _load_or_compute_embeddings(self):
        if os.path.exists(self.embedding_file):
            print(f"[INFO] Loading existing fact embeddings from {self.embedding_file}")
            return torch.load(self.embedding_file, map_location=self.device)
        
        print("[INFO] Computing new embeddings for 'fact' field...")
        # 仅抽取 fact 字段，防空处理
        corpus_texts = [doc.get('fact', '') or "" for doc in self.corpus]
        corpus_embeddings = self.embedder.encode(
            corpus_texts, 
            show_progress_bar=True, 
            normalize_embeddings=True,
            batch_size=self.batch_size
        )
        torch.save(corpus_embeddings, self.embedding_file)
        return corpus_embeddings

    def search(self, query: str, num: int):
        if not query.strip(): 
            return [], []
        query_embedding = self.embedder.encode(query, normalize_embeddings=True)
        hits = semantic_search(query_embedding, self.corpus_embeddings, top_k=num)[0]
        
        results, scores = [], []
        for hit in hits:
            results.append(self.corpus[hit['corpus_id']])
            scores.append(hit['score'])
        return results, scores


class ReasonBM25Retriever:
    """仅针对 reason 字段进行分词和检索"""
    def __init__(self, config):
        if hasattr(config, "dictionary_path") and config.dictionary_path:
            lawa.load_userdict(config.dictionary_path)
            
        self.corpus = load_corpus(config.corpus_path)
        
        # 仅抽取 reason 字段建库
        self.docs_raw = [doc.get("reason", "") or "" for doc in self.corpus]
        print("[INFO] Tokenizing 'reason' field for BM25...")
        self.docs_tokenized = [list(lawa.cut(text)) for text in self.docs_raw]
        self.bm25 = BM25Okapi(self.docs_tokenized, k1=1.5, b=0.5)

    def search(self, query: str, num: int):
        if not query.strip():
            return [], []
        
        # 符号清洗
        query = re.sub(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?/\'"、。，！？；：「」『』（）【】《》﹁﹂﹃﹄‘’“”～﹏丶]', ' ', query)
        query_tokens = list(lawa.cut(query))
        
        scores = self.bm25.get_scores(query_tokens)
        if len(scores) == 0:
            return [], []

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:num]
        doc_ids = [idx for idx, _ in ranked]
        top_scores = [score for _, score in ranked]
        results = load_docs(self.corpus, doc_ids)
        
        return results, top_scores

# ==========================================
# 3. 核心：类案混合检索器 (三路融合 + 权威性 + MMR)
# ==========================================

class SimilarCaseRetriever:
    def __init__(self, config):
        self.t2v_retriever = FactText2vecRetriever(config,f"cuda:{config.gpu_ids}")
        self.bm25_retriever = ReasonBM25Retriever(config)
        
        self.topk = config.topk
        self.search_depth = config.search_depth
        
        # 权重设置
        self.w_charge = 0.5   # 罪名软加权最高
        self.w_t2v = 0.3      # 事实语义次之
        self.w_bm25 = 0.2     # 情节匹配再次
        
        # 权威性加成表
        self.court_boost = {
            1: 1.15, 
            2: 1.10, 
            3: 1.05, 
            4: 1.00
        }
        
        # MMR 多样性权重 (lambda): 越高越看重原相关性，越低越看重多样性
        self.mmr_lambda = 0.7 

    def _min_max_norm(self, scores):
        if not scores: return []
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s: return [0.0 for _ in scores]
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def _mmr_rerank(self, candidates: List[Dict], scores: List[float], top_k: int) -> Tuple[List[Dict], List[float]]:
        """基于 psi_score 动态归一化的 MMR 多样性重排"""
        if not candidates: return [], []
        
        # 提取 psi_score (容错处理)
        psi_scores = []
        for doc in candidates:
            try:
                psi = float(doc.get("psi_score", 0.0))
            except:
                psi = 0.0
            psi_scores.append(psi)
            
        d_max = max(psi_scores) - min(psi_scores)
        if d_max == 0: d_max = 1e-9  # 防止除以0
        
        selected_indices = []
        unselected_indices = list(range(len(candidates)))
        
        while len(selected_indices) < top_k and unselected_indices:
            if not selected_indices:
                # 第一轮直接选得分最高的
                best_idx = unselected_indices[0]
                best_idx_in_unselected = 0
                for i, idx in enumerate(unselected_indices):
                    if scores[idx] > scores[best_idx]:
                        best_idx = idx
                        best_idx_in_unselected = i
                selected_indices.append(best_idx)
                unselected_indices.pop(best_idx_in_unselected)
            else:
                # 后续使用 MMR 公式
                best_mmr = -float('inf')
                best_idx = -1
                best_idx_in_unselected = -1
                
                for i, idx in enumerate(unselected_indices):
                    # 计算该候选与已选中集合的最大相似度惩罚
                    max_sim = 0.0
                    for s_idx in selected_indices:
                        sim = 1.0 - (abs(psi_scores[idx] - psi_scores[s_idx]) / d_max)
                        if sim > max_sim:
                            max_sim = sim
                    
                    # MMR 核心公式
                    mmr_score = self.mmr_lambda * scores[idx] - (1 - self.mmr_lambda) * max_sim
                    
                    if mmr_score > best_mmr:
                        best_mmr = mmr_score
                        best_idx = idx
                        best_idx_in_unselected = i
                        
                selected_indices.append(best_idx)
                unselected_indices.pop(best_idx_in_unselected)
                
        final_docs = [candidates[i] for i in selected_indices]
        final_scores = [scores[i] for i in selected_indices]
        return final_docs, final_scores


    def search(self, fact_query: str, charge_query: List[str], reason_query: str, num: int = None):
        target_k = num if num else self.topk
        candidate_k = target_k * self.search_depth  # 扩大候选池
        
        start_time = time.time()

        # 1. 双路召回
        bm25_docs, bm25_scores = self.bm25_retriever.search(reason_query, candidate_k)
        t2v_docs, t2v_scores = self.t2v_retriever.search(fact_query, candidate_k)

        # 2. 构建去重候选池
        pool = {}  
        def add_pool(docs, scores, key):
            for doc, sc in zip(docs, scores):
                doc_id = doc.get("id") or doc.get("pid") or id(doc)
                if doc_id not in pool:
                    pool[doc_id] = {"doc": doc, "bm25": 0.0, "t2v": 0.0}
                pool[doc_id][key] = sc

        add_pool(bm25_docs, bm25_scores, "bm25")
        add_pool(t2v_docs, t2v_scores, "t2v")

        # 3. 分数提取与归一化
        bm25_all = [v["bm25"] for v in pool.values()]
        t2v_all = [v["t2v"] for v in pool.values()]
        bm25_norm = self._min_max_norm(bm25_all)
        t2v_norm = self._min_max_norm(t2v_all)

        # 4. 融合、加权与权威性增益
        for (doc_id, info), bn, tn in zip(pool.items(), bm25_norm, t2v_norm):
            doc = info["doc"]
            
            # ==========================================
            # 【核心修改】：软匹配罪名 (Jaccard 重合度计算)
            # ==========================================
            doc_charge_list = doc.get("charge", [])
            # 防错：如果原数据里的 charge 是字符串，转成列表
            if not isinstance(doc_charge_list, list): 
                doc_charge_list = [doc_charge_list] if doc_charge_list else []
            
            set_query = set(charge_query)
            set_doc = set(doc_charge_list)
            
            # 计算交集与并集
            if not set_query and not set_doc:
                charge_score = 0.0  # 都没有罪名记录
            elif not set_query or not set_doc:
                charge_score = 0.0  # 其中一方为空
            else:
                intersection = set_query.intersection(set_doc)
                union = set_query.union(set_doc)
                charge_score = len(intersection) / len(union)  # 算出 0.0 到 1.0 的重合度比例
                charge_score = charge_score ** 5
            
            # 基础融合分
            hybrid_score = (self.w_charge * charge_score) + (self.w_t2v * tn) + (self.w_bm25 * bn)
            
            # --- 权威性加成 (Court Level Boosting) ---
            c_level = doc.get("court_level", 4)
            boost = self.court_boost.get(c_level, 1.0)
            
            info["final_score"] = hybrid_score * boost

        # 按融合分数倒排，截取供 MMR 使用的池子
        ranked_pool = sorted(pool.items(), key=lambda x: x[1]["final_score"], reverse=True)[:candidate_k]
        
        candidate_docs = [info["doc"] for _, info in ranked_pool]
        candidate_scores = [info["final_score"] for _, info in ranked_pool]

        # 5. MMR 多样性重排
        final_docs, final_scores = self._mmr_rerank(candidate_docs, candidate_scores, target_k)

        end_time = time.time()
        print(f"[DEBUG] 类案检索完成，耗时: {end_time - start_time:.4f}s")
        
        return final_docs, final_scores
# ==========================================
# 4. FastAPI 服务端
# ==========================================

app = FastAPI()

# 严格按照用户提供的 JSON 格式接收请求
class CaseQueryItem(BaseModel):
    search_type: str = Field(alias="检索类型", default="类案检索")
    fact_query: str = Field(alias="检索案情", default="")
    charge: List[str] = Field(alias="罪名", default_factory=list) 
    other_reason: str = Field(alias="其他情节", default="")

class CaseQueryRequest(BaseModel):
    query: CaseQueryItem
    topk: Optional[int] = None

# 占位全局变量
retriever = None

@app.post("/retrieve_case")
def retrieve_case_endpoint(request: CaseQueryRequest):
    req_topk = request.topk if request.topk else retriever.topk
    
    # 提取三大要素
    fact_q = request.query.fact_query
    charge_q = request.query.charge
    reason_q = request.query.other_reason
    
    docs, scores = retriever.search(fact_query=fact_q, charge_query=charge_q, reason_query=reason_q, num=req_topk)
    
    # 构建返回值
    resp = []
    for doc, score in zip(docs, scores):
        resp.append({
            "score": round(score, 4),
            "pid": doc.get("pid", ""),
            "charge": doc.get("charge", []),
            "court_level": doc.get("court_level", 4),
            "psi_score": doc.get("psi_score", 0),
            "fact": doc.get("fact", ""),
            "reason": doc.get("reason", ""),
            "result": doc.get("result", "")
        })
        
    return {"results": resp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Similar Case Retriever.")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to lecard_court_psi.jsonl")
    parser.add_argument("--retriever_model", type=str, default="shibing624/text2vec-base-chinese-paraphrase")
    parser.add_argument("--dictionary_path", type=str, default='', help="jieba dictionary for law")
    parser.add_argument("--topk", type=int, default=5, help="Number of final results.")
    parser.add_argument("--search_depth", type=int, default=10, help="Recall multiplier for MMR pool.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--port", type=int, default=8007)
    parser.add_argument("--gpu_ids", type=int, default=0, help="GPU device IDs to use.")
    args = parser.parse_args()


    cfg = Config(
        corpus_path=args.corpus_path,
        retrieval_model_path=args.retriever_model,
        dictionary_path=args.dictionary_path,
        retrieval_batch_size=args.batch_size,
        topk=args.topk,
        search_depth=args.search_depth
    )

    retriever = SimilarCaseRetriever(cfg)
    uvicorn.run(app, host="0.0.0.0", port=args.port)