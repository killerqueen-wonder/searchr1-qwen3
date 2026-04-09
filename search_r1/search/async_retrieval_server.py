import json
import os
import re
import time
import asyncio
import httpx
import argparse
from typing import List, Dict, Optional, Tuple

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

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
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

'''Start an Async RAG system powered by vLLM'''

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
                pass
    print(f"[INFO] 成功加载 {len(corpus)} 条数据。")
    return corpus

def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

# ==========================================
# 核心改造 1：全局异步大模型客户端 (取代 SharedLLM)
# ==========================================
class AsyncVLLMClient:
    def __init__(self, config):
        self.vllm_url = getattr(config, "vllm_url", "http://127.0.0.1:8007/v1/completions")
        self.model_name = getattr(config, "filter_model", "Qwen3-8B")
        self.client = httpx.AsyncClient(timeout=60.0)
        print(f"[INFO] Async vLLM Client pointing to {self.vllm_url} (Model: {self.model_name})")

    async def generate_async(self, prompt: str, max_new_tokens: int = 64) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "stop": ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]
        }

        max_retries = 3  # 最大重试次数
        base_wait_time = 1.5  # 基础等待时间（秒）

        for attempt in range(max_retries):
            try:
                response = await self.client.post(self.vllm_url, json=payload)
                
                # 如果遇到 503 忙碌，主动拦截并等待重试
                if response.status_code == 503:
                    print(f"[WARNING] vLLM队列已满 (503), 正在重试 ({attempt+1}/{max_retries})...")
                    await asyncio.sleep(base_wait_time * (attempt + 1)) # 每次重试等待时间递增
                    continue
                    
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["text"].strip()
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[ERROR] vLLM 最终生成失败 (已重试 {max_retries} 次): {e}")
                    return ""
                
                # 遇到其他网络波动也稍微等一下重试
                await asyncio.sleep(base_wait_time)
                
        return ""

    async def close(self):
        await self.client.aclose()


# ==========================================
# 检索器基类与其他检索器 (内部逻辑不变，供外部异步调用)
# ==========================================

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


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool,context: Optional[List[str]] = None):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool,context: Optional[List[str]] = None):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False,context: Optional[List[str]] = None):
        return self._search(query, num, return_score,context)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False,context: Optional[List[str]] = None):
        return self._batch_search(query_list, num, return_score,context)


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
            m2 = re.search(r"(?<=关于)\s*(.*?)\s*(?=的)", text, flags=re.S)

            if m1:
                law_name = m1.group(1).strip()
            elif m2:#匹配法律解释
                law_name = m2.group(1).strip()

            # 提取 “法条编号”：从第一次出现“\n  第”，到之后第一次出现“条\n”
            article_id = ""
            m3 = re.search(r"\n\s*(第.*?条)\n", text, flags=re.S)
            if m3:
                article_id = f"{m3.group(1)}"

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
                weighted_tokens.extend([article_id] * (weight_factor))

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
        self.bm25 = BM25Okapi(self.docs_tokenized,k1=1.5, b=0.3)

        
        self.max_process_num = 8

    def _search(self, query: str, num: int = None, return_score: bool = False,context: Optional[List[str]] = None):
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
                return str(num)
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

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False,context: Optional[List[str]] = None):
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


class Text2vecRetriever(BaseRetriever):
    def __init__(self, config, device: str = None):
        super().__init__(config)
        
        # 无论物理 ID 是多少，对当前进程第一张卡永远是 cuda:0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"[INFO] Text2vec model strictly locked on: {self.device}")
        self.embedder = SentenceModel(config.retrieval_model_path, device=self.device)
        
        self.corpus = load_corpus(self.corpus_path)
        self.embedding_file = self._get_embedding_filename(self.corpus_path)
        self.corpus_embeddings = self._load_or_compute_embeddings()
        
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _get_embedding_filename(self, jsonl_filename):
        base_name = os.path.splitext(os.path.basename(jsonl_filename))[0]
        dir_name = os.path.dirname(jsonl_filename)
        return os.path.join(dir_name, f"{base_name}_embeddings.pt")

    def _load_or_compute_embeddings(self):
        if os.path.exists(self.embedding_file):
            return torch.load(self.embedding_file, map_location=self.device)
        return self._compute_and_save_embeddings()

    def _compute_and_save_embeddings(self):
        # 保持 content 完整性
        corpus_texts = [doc.get('contents') or doc.get('text') or doc.get('content') or str(doc) for doc in self.corpus]
        # 禁用多进程以防止显存碎片化
        corpus_embeddings = self.embedder.encode(
            corpus_texts, 
            show_progress_bar=True, 
            normalize_embeddings=True,
            batch_size=self.batch_size
        )
        torch.save(corpus_embeddings, self.embedding_file)
        return corpus_embeddings

    def _search(self, query: str, num: int = None, return_score: bool = False,context: Optional[List[str]] = None):
        if num is None: num = self.topk
        if not query.strip(): return ([], []) if return_score else []
        
        query_embedding = self.embedder.encode(query, normalize_embeddings=True)
        hits = semantic_search(query_embedding, self.corpus_embeddings, top_k=num)[0]
        
        results, scores = [], []
        for hit in hits:
            results.append(self.corpus[hit['corpus_id']])
            scores.append(hit['score'])
        return (results, scores) if return_score else results
    
    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False,context: Optional[List[str]] = None):
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
        self.bm25_retriever = BM25WeightRetriever(config)
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

    def _search(self, query: str, num: int = None, return_score: bool = False,context: Optional[List[str]] = None):
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

    def _batch_search(self, query_list, num=None, return_score=False,context: Optional[List[str]] = None):
        results = []
        scores = []
        for q in query_list:
            r, s = self._search(q, num, True)
            results.append(r)
            scores.append(s)
        return (results, scores) if return_score else results


    
class HybridFilterRetriever(HybridRetriever):
    def __init__(self, config):
        """
        初始化混合过滤检索器 (HybridFilterRetriever)。
        
        该类实现了“粗排+精排”的两阶段检索策略：
        1. 粗排阶段：并行调用 BM25 权重检索器和 Text2vec 语义检索器，并进行分数融合。
        2. 精排阶段：将粗排结果送入大语言模型 (LLM)，根据检索词和语境 (context) 进行意图对齐筛选。

        Args:
            config (Config): 全局配置对象，必须包含以下字段：
                - retrieval_topk (int): 最终输出给用户的文档数量。
                - search_depth (int): 混合检索的深度系数，用于粗排阶段计算候选池倍数。
                - bm25_weight (float): 混合检索中 BM25 分数的权重（Text2vec 权重固定为 10）。
                - filter_model (str): 用于过滤任务的 LLM 模型路径或名称。
                - gpu_memory_limit_per_gpu (list/int): 每张显卡的内存限制（单位: GB）。
                - retrieval_model_path (str): Text2vec 嵌入模型的路径。
                - corpus_path (str): 语料库本地 JSONL 文件路径。

        Note:
            显存策略：为了在 2 张 V100 上优化性能，该构造函数将 Text2vec 锁定在 cuda:0，
            并为 LLM 预留了分片加载空间。通过 max_memory 映射，手动在 0 号卡上预留了 3GB 
            空间给系统和嵌入模型，以防止长文本推理时发生 OOM。
        """
        retrieval_device = f"cuda:{config.gpu_ids}"
        
        self.bm25_retriever = BM25WeightRetriever(config)
        self.text2vec_retriever = Text2vecRetriever(config, device=retrieval_device)
        
        self.topk = config.retrieval_topk
        self.search_depth = config.search_depth
        self.w_bm25 = config.bm25_weight
        self.w_t2v = 10
        self.candidate_k = self.topk * self.search_depth
        

        self.prompt_template = (
            "### 任务指令\n"
            "你是一名资深的法律文书核查员。请根据提供的【语境信息】和【检索词】，从【备选文本】中筛选出语义最相关且完全符合法条要求的文档编号。\n\n"
            "### 核心规则（必须遵守）：\n"
            "1. **语境优先**：如果【语境信息】提到了具体的法律名称或条文编号（如：第二条），所选文本必须与之匹配。若无匹配项，直接返回 []。\n"
            "2. **严禁凑数**：筛选结果是不超过 {topk} 段。如果只有 1 段符合，仅返回该段编号；如果没有符合的，返回 []。严禁为了达到 {topk} 个而返回无关文本。\n"
            "3. **输出格式**：只允许输出一个 Python 格式的整数列表，例如：[1] 或 [1, 2]。\n"
            "4. **禁止解释**：严禁输出任何分析过程、思考内容（<think>）或多余的文字说明。\n\n"
            "5. **索引对齐**：请仔细阅读每段备选文本的内容。如果【语境信息】要求‘第二条’，请找到内容中确实描述‘第二条’的文本，并输出该文本对应的【编号】。"
            "### 输入数据\n"
            "- 【检索词】：{query}\n"
            "- 【语境信息】：{context}\n"
            "- 【备选文本】：\n{results}\n\n"
            "### 筛选结果（仅输出列表）："
        )

    def _llm_filter(self,num:int, query: str, candidates: List[Dict], scores: List[float], 
                    context: str = None) -> Tuple[List[Dict], List[float]]:
        if not candidates: return [], []

        # 保持 content 完整以供筛选
        # candidates_data = [{"content": doc.get("content", "")} for doc in candidates]
        # results_str = json.dumps(candidates_data, ensure_ascii=False)
        
        # 显式为每一条候选文档添加序号标签
        candidates_formatted = []
        for i, doc in enumerate(candidates, 1):
            content = doc.get("content", "").replace('\n', ' ')
            candidates_formatted.append(f"编号 {i}: {content}")
        
        # 将列表转为带换行的字符串，方便模型阅读
        results_str = "\n".join(candidates_formatted)
        prompt = self.prompt_template.format(results=results_str, query=query, 
                                             topk=num,context=context if context else "无")
        print(f'[debug]{prompt}')


        global shared_llm
        if not shared_llm or not shared_llm.model:
            print("[WARNING] LLM filter triggered but model not loaded. Skipping filter.")
            return candidates[:self.topk], scores[:self.topk]

        response = shared_llm.generate(prompt, max_new_tokens=64)
        text_num = self._extract_numbers(response)
        print(f'[debug]response:{response}')
        print(f'[debug]text_num:{text_num}')
        
        if not text_num: 
            return [{'content':'检索不到相关内容，请尝试修改检索词或搜索其他方向。'}], [0]

        filtered_results = [candidates[i-1] for i in text_num if i-1 < len(candidates)]
        filtered_scores = [scores[i-1] for i in text_num if i-1 < len(scores)]
        return filtered_results[:self.topk], filtered_scores[:self.topk]
        


    def _extract_numbers(self, text):
        pattern = r'\[([\d,\s]+)\]'
        match = re.search(pattern, text)
        if match:
            nums = re.findall(r'\d+', match.group(1))
            return [int(n) for n in nums]
        return None
    
    def _search(self, query: str, num: int = None, return_score: bool = False, context: str = None):
        # 1. 使用父类 HybridRetriever 进行初步检索 topk * search_depth个
        if not num:
            num=self.topk
        
        
        candidates, scores = super()._search(query, num*self.search_depth, return_score=True)
        
        start_time = time.time()
        
        # 2. 使用 LLM 进行过滤 (Precision)
        filtered_results, filtered_scores = self._llm_filter(query,num, candidates, scores, context)
        
        end_time = time.time()
        print(f"[DEBUG] LLM Filter time: {end_time - start_time:.4f} s, Input: {len(candidates)} -> Output: {len(filtered_results)}")

        if return_score:
            return filtered_results, filtered_scores
        return filtered_results
    
    
    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False, context_list: Optional[List[str]] = None):
        results = []
        scores = []
        
        # 如果没传 context，则补齐为 None 列表
        if context_list is None:
            context_list = [None] * len(query_list)
            
        # 一一对应分发
        for q, ctx in zip(query_list, context_list):
            r, s = self._search(q, num, True, context=ctx) # 传入单条 context
            results.append(r)
            scores.append(s)
        return (results, scores) if return_score else results



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
# 类案混合检索器 (三路融合 + 权威性 + MMR)
# ==========================================

class SimilarCaseRetriever:
    def __init__(self, config):
        self.t2v_retriever = FactText2vecRetriever(config,f"cuda:{config.gpu_ids}")
        self.bm25_retriever = ReasonBM25Retriever(config)
        
        self.topk = config.retrieval_topk
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
# 核心改造 2：异步化的过滤检索器
# ==========================================
class AsyncHybridFilterRetriever(HybridRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.prompt_template = (
            "### 任务指令\n"
            "你是一名资深的法律文书核查员。请根据提供的【语境信息】和【检索词】，从【备选文本】中筛选出语义最相关且完全符合法条要求的文档编号。\n\n"
            "### 核心规则（必须遵守）：\n"
            "1. **语境优先**：如果【语境信息】提到了具体的法律名称或条文编号（如：第二条），所选文本必须与之匹配。若无匹配项，直接返回 []。\n"
            "2. **严禁凑数**：筛选结果是不超过 {topk} 段。如果只有 1 段符合，仅返回该段编号；如果没有符合的，返回 []。严禁为了达到 {topk} 个而返回无关文本。\n"
            "3. **输出格式**：只允许输出一个 Python 格式的整数列表，例如：[1] 或 [1, 2]。\n"
            "4. **禁止解释**：严禁输出任何分析过程、思考内容（<think>）或多余的文字说明。\n\n"
            "5. **索引对齐**：请仔细阅读每段备选文本的内容。如果【语境信息】要求‘第二条’，请找到内容中确实描述‘第二条’的文本，并输出该文本对应的【编号】。"
            "### 输入数据\n"
            "- 【检索词】：{query}\n"
            "- 【语境信息】：{context}\n"
            "- 【备选文本】：\n{results}\n\n"
            "### 筛选结果（仅输出列表）："
        )

    async def _async_llm_filter(self, num: int, query: str, candidates: List[Dict], scores: List[float], 
                                context: str = None) -> Tuple[List[Dict], List[float]]:
        if not candidates: return [], []

        candidates_formatted = []
        for i, doc in enumerate(candidates, 1):
            content = doc.get("content", "").replace('\n', ' ')
            candidates_formatted.append(f"编号 {i}: {content}")
        
        results_str = "\n".join(candidates_formatted)
        prompt = self.prompt_template.format(results=results_str, query=query, topk=num, context=context if context else "无")

        global async_vllm_client
        if not async_vllm_client:
            return candidates[:self.topk], scores[:self.topk]

        # 【非阻塞调用】
        response = await async_vllm_client.generate_async(prompt, max_new_tokens=64)
        text_num = self._extract_numbers(response)
        
        if not text_num: 
            return [{'content':'检索不到相关内容，请尝试修改检索词或搜索其他方向。'}], [0]

        filtered_results = [candidates[i-1] for i in text_num if i-1 < len(candidates)]
        filtered_scores = [scores[i-1] for i in text_num if i-1 < len(scores)]
        return filtered_results[:self.topk], filtered_scores[:self.topk]

    def _extract_numbers(self, text):
        pattern = r'\[([\d,\s]+)\]'
        match = re.search(pattern, text)
        if match:
            nums = re.findall(r'\d+', match.group(1))
            return [int(n) for n in nums]
        return None
    
    async def async_search(self, query: str, num: int = None, return_score: bool = False, context: str = None):
        if not num: num = self.topk
        # 1. 将密集的打分计算丢到后台线程，避免阻塞 FastAPI 事件循环
        async with get_gpu_semaphore():
            candidates, scores = await asyncio.to_thread(super()._search, query, num * self.search_depth, True)
        
        # 2. 异步精排
        filtered_results, filtered_scores = await self._async_llm_filter(num, query, candidates, scores, context)
        
        if return_score:
            return filtered_results, filtered_scores
        return filtered_results
    
    async def async_batch_search(self, query_list: List[str], num: int = None, return_score: bool = False, context_list: Optional[List[str]] = None):
        if context_list is None: context_list = [None] * len(query_list)
        # 并发执行所有 query
        tasks = [self.async_search(q, num, True, ctx) for q, ctx in zip(query_list, context_list)]
        batch_res = await asyncio.gather(*tasks)
        
        results, scores = [], []
        for r, s in batch_res:
            results.append(r)
            scores.append(s)
            
        return (results, scores) if return_score else results


def get_async_retriever(config):
    if config.retrieval_method == "hybrid":
        return HybridRetriever(config)
    elif config.retrieval_method == "hybrid_filter":
        return AsyncHybridFilterRetriever(config)
    else:
        print("[error]retriever name error.")


# ==========================================
# FastAPI & Models
# ==========================================
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
        port =8006,#port端口号
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        dictionary_path:str="",
        search_depth:int =5,
        bm25_weight:int=10,
        bm25_weight_factor:int =3,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.5,

        filter_model:str="",
        **kwargs
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.case_corpus_path = kwargs.get("case_corpus_path", "")
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
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        self.filter_model=filter_model

        self.__dict__.update(kwargs)
       

class UnifiedQueryItem(BaseModel):
    search_type: str = Field(alias="检索类型")
    fact_query: str = Field(alias="检索案情", default="")
    charge: List[str] = Field(alias="罪名", default_factory=list)
    other_reason: str = Field(alias="其他情节", default="")
    keywords: str = Field(alias="关键词", default="")
    search_purpose: str = Field(alias="检索目的", default="")

class UnifiedQueryRequest(BaseModel):
    query: UnifiedQueryItem
    topk: Optional[int] = 5



# 【修复关键】：把解析参数和初始化的代码移到全局作用域
parser = argparse.ArgumentParser(description="Unified Law and Case Retriever.")
parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
parser.add_argument("--corpus_path", type=str, required=True, help="法条语料路径")
parser.add_argument("--case_corpus_path", type=str, required=True, help="类案语料路径")
parser.add_argument("--dictionary_path", type=str, default='', help="jieba dictionary for law")
parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
parser.add_argument("--search_depth", type=int, default=5, help="hydrid search depth")
parser.add_argument("--bm25_weight", type=int, default=10)
parser.add_argument("--bm25_weight_factor", type=int, default=3)
parser.add_argument("--bm25_k1", type=float, default=1.5, help="BM25 k1 parameter")
parser.add_argument("--bm25_b", type=float, default=0.5, help="BM25 b parameter")

parser.add_argument("--retriever_name", type=str, default="text2vec", help="Name of the retriever model.")
parser.add_argument("--retriever_model", type=str, default="shibing624/text2vec-base-chinese-paraphrase", help="Path of the retriever model.")
parser.add_argument("--filter_model", type=str, default="Qwen3-8B")
parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')

parser.add_argument("--port", type=int, default=8006, help="the API port")
parser.add_argument("--gpu_ids", type=int,  default=2, help="GPU device IDs to use.")
parser.add_argument("--gpu_memory_limit_per_gpu", type=int, nargs='+', default=[18], help="GPU memory limit per GPU in GB.")

# 使用 parse_known_args 防止与 Uvicorn 的内部参数冲突
args, unknown = parser.parse_known_args()

# 全局初始化 Config
global_config = Config(
    retrieval_method=args.retriever_name,               
    retrieval_model_path=args.retriever_model,          
    index_path=args.index_path,
    corpus_path=args.corpus_path,
    case_corpus_path=args.case_corpus_path,
    retrieval_topk=args.topk,
    faiss_gpu=args.faiss_gpu,
    port=args.port,  
    gpu_ids=args.gpu_ids,  
    gpu_memory_limit_per_gpu=args.gpu_memory_limit_per_gpu,
    retrieval_pooling_method="mean",
    retrieval_query_max_length=256,
    retrieval_use_fp16=True,
    retrieval_batch_size=512,
    dictionary_path=args.dictionary_path,
    search_depth=args.search_depth,
    bm25_weight=args.bm25_weight,
    bm25_weight_factor=args.bm25_weight_factor,
    bm25_k1=args.bm25_k1,
    bm25_b=args.bm25_b,
    filter_model=args.filter_model,
    vllm_url="http://127.0.0.1:8007/v1/completions" 
)

# 声明全局变量，但【不在这里实例化模型】！
law_retriever = None
case_retriever = None
async_vllm_client = None
gpu_semaphore = None

def get_gpu_semaphore():
    """懒加载获取 GPU 信号量，确保在当前事件循环中初始化"""
    global gpu_semaphore
    if gpu_semaphore is None:
        # 这里设置为你的并发限制
        gpu_semaphore = asyncio.Semaphore(4)
    return gpu_semaphore

app = FastAPI()

# ==========================================
# FastAPI 生命周期与路由
# ==========================================
@app.on_event("startup")
async def startup_event():
    global law_retriever, case_retriever, async_vllm_client
    import os
    pid = os.getpid()
    print(f"[INFO] Worker {pid} 正在启动并独立加载模型到显存...")
    

    # 1. 启动 vLLM 客户端
    async_vllm_client = AsyncVLLMClient(global_config)
    
    # 2. 独立加载法律检索器
    law_retriever = get_async_retriever(global_config)
    
    # 3. 独立加载类案检索器
    case_config = Config(**global_config.__dict__)
    case_config.corpus_path = global_config.case_corpus_path
    case_retriever = SimilarCaseRetriever(case_config)
    
    print(f"[INFO] Worker {pid} 模型加载完毕，可以开始接客了！")

@app.on_event("shutdown")
async def shutdown_event():
    global async_vllm_client
    if async_vllm_client:
        await async_vllm_client.close()

# ==========================================
# 核心改造 3：全异步接口路由
# ==========================================
@app.post("/retrieve")
async def unified_retrieve_endpoint(request: UnifiedQueryRequest):
    # 移除了 inference_lock，改为异步执行
    print(f"[INFO] 收到新请求，正在异步并发处理中...")
    start_time_total = time.time()
    req_type = request.query.search_type
    req_topk = request.topk
    
    # ==========================================
    # 分支一：类案检索
    # ==========================================
    if req_type == "类案检索":
        fact_q = request.query.fact_query
        charge_q = request.query.charge
        reason_q = request.query.other_reason
        
        search_k = req_topk  # 类案只总结不筛选
        

        # 将 CPU 密集的检索丢到后台线程，不阻塞主事件循环
        async with get_gpu_semaphore():    
            docs, scores = await asyncio.to_thread(
                case_retriever.search,
                fact_query=fact_q, 
                charge_query=charge_q, 
                reason_query=reason_q, 
                num=search_k
            )
        
        # 定义一个异步获取单个摘要的闭包
        async def fetch_summary(doc):
            doc_fact = doc.get("fact", "")
            doc_reason = doc.get("reason", "")
            doc_result = doc.get("result", "")
            
            prompt = (
                "### 任务指令\n"
                "你是一名资深的法官助理。请仔细对比用户的【检索案情】与检索到的【候选案例】。\n"
                "请结合候选案例的“案情”、“裁判推理”和“判决结果”，写一份200字左右的案例简报。\n\n"
                "### 简报撰写要求：\n"
                "1. 明确指出该案例与【检索案情】的相似之处。\n"
                "2. 简明扼要地概括法院的裁判推理（尤其是对关键情节的认定逻辑）。\n"
                "3. 清楚写明最终的判决结论。\n\n"
                "### 输入数据\n"
                f"- 【检索案情】：{fact_q}\n"
                f"- 【候选案例 - 案情】：{doc_fact[:1800]}...\n"
                f"- 【候选案例 - 裁判推理】：{doc_reason[:1800]}...\n"
                f"- 【候选案例 - 判决结果】：{doc_result[:1400]}...\n\n"
                "### 输出（请直接输出200字简报）："
            )
            analysis = await async_vllm_client.generate_async(prompt, max_new_tokens=400)
            return analysis.strip()

        # 并发请求所有候选项的摘要
        tasks = [fetch_summary(doc) for doc in docs[:req_topk]]
        analyses = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_resp = []
        llm_summaries = []
        valid_count = 0
        
        # 组装返回结果（完全还原原版格式和逻辑）
        for i, (doc, score) in enumerate(zip(docs[:req_topk], scores[:req_topk])):
            analysis = analyses[i]
            if isinstance(analysis, Exception):
                print(f"[ERROR] LLM processing failed for doc {i}: {analysis}")
                continue
                
            doc_fact = doc.get("fact", "")
            doc_reason = doc.get("reason", "")
            doc_result = doc.get("result", "")
            
            doc_record = {
                "score": round(score, 4),
                "pid": doc.get("pid", ""),
                "charge": doc.get("charge", []),
                "court_level": doc.get("court_level", 4),
                "psi_score": doc.get("psi_score", 0),
                "fact": doc_fact,
                "reason": doc_reason,
                "result": doc_result,
                "llm_summary": analysis 
            }
            final_resp.append(doc_record)
            llm_summaries.append(f"【参考类案 {valid_count + 1}】\n{analysis}")
            valid_count += 1

        if not llm_summaries:
            overall_summary = "检索完毕。未发现与您输入案情高度相似的典型案例。"
        else:
            overall_summary = "\n\n".join(llm_summaries)
            
        return {
            "检索类型": "类案检索",
            "llm_summary": overall_summary,
            "results": final_resp
        }

    # ==========================================
    # 分支二：法律检索 (法条)
    # ==========================================
    elif req_type == "法律检索":
        keywords = request.query.keywords
        context = request.query.search_purpose
        
        # 适配异步 Filter 检索器或普通检索器
        if isinstance(law_retriever, AsyncHybridFilterRetriever):
            async with get_gpu_semaphore(): 
                results, scores = await law_retriever.async_batch_search(
                    query_list=[keywords],
                    num=req_topk,
                    return_score=True,
                    context_list=[context] if context else [None]
                )
        else:
            async with get_gpu_semaphore(): 
                results, scores = await asyncio.to_thread(
                    law_retriever.batch_search,
                    query_list=[keywords],
                    num=req_topk,
                    return_score=True,
                    context=[context] if context else [None]
                )
        
        resp = []
        for i, single_result in enumerate(results):
            processed_docs = []
            for j, doc in enumerate(single_result):
                standard_doc = doc.copy()
                standard_doc['content'] = doc.get('content') or doc.get('contents') or doc.get('text') or ""
                processed_docs.append({"document": standard_doc, "score": scores[i][j]})
            resp.append(processed_docs)
            
        return {
            "检索类型": "法律检索",
            "result": resp[0]  # 因为 query_list 只有一个元素，所以取 [0]
        }
    
    # ==========================================
    # 分支三：容错处理
    # ==========================================
    else:
        return {"error": f"未知的检索类型：'{req_type}'，请使用'类案检索'或'法律检索'。"}
    
if __name__ == "__main__":
    print("[INFO] Async Unified Retriever Service Started Successfully!")

    uvicorn.run("async_retrieval_server:app", host="0.0.0.0", port=global_config.port, workers=1)
    