import argparse
import glob
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import faiss
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


BASE_DIR = os.environ.get(
    "BASE_DIR",
    "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1",
)
LOG_DIR = os.path.join(BASE_DIR, "dataset", "result", "bench_result", "log")
os.makedirs(LOG_DIR, exist_ok=True)

current_time = time.strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(LOG_DIR, f"autorefine_infer_pipeline_{current_time}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


FALLBACK_PROMPTS = {
    "first_gen_answer": (
        "根据以下提供的文档内容撰写一篇笔记。笔记应整合所有能够帮助回答指定问题的相关原文信息，"
        "并形成一段连贯的文本。请确保笔记包含所有对回答问题有用的原文信息。\n\n"
        "需要回答的问题： {query}\n文档内容： {refs}\n\n请提供你撰写的笔记："
    ),
    "gen_new_query": (
        "任务：根据笔记，提出两个新问题。这些新问题将用于检索文档，以补充笔记并帮助回答原始问题。"
        "新问题应简明扼要，并包括有利于检索的关键词。新问题应避免与现有问题列表重复。\n\n"
        "原始问题：{query}\n笔记：{note}\n\n现有问题列表：{query_log}\n\n提供两个新问题："
    ),
    "refine_note": (
        "任务：根据检索到的文档，用尚未包含但对回答问题有用的内容补充笔记。"
        "补充内容应使用检索到的文档中的原始文本，并尽可能多的包含检索到的文档的信息。\n\n"
        "问题：{query}\n检索到的文档：{refs}\n\n笔记：{note}\n\n提供补充后的笔记："
    ),
    "compare": (
        "任务：请你帮我判断提供的哪个笔记更好，具体评价标准如下：\n"
        "1. 包含与问题直接相关的关键信息。\n"
        "2. 信息的全面性：是否涵盖了所有相关的方面和细节。\n"
        "3. 信息的细节程度：是否提供了足够的细节来深入理解问题。\n"
        "4. 实用性：笔记是否提供了实际的帮助和解决方案。\n\n"
        "请严格按照以下要求进行判断：\n"
        "- 如果笔记2没有在笔记1的基础上增加新的有意义的内容，或者只是增加了多余的信息，"
        "请直接返回 {{\"status\":\"False\"}}。\n"
        "- 如果笔记2比笔记1在上述标准上有明显改进，请直接返回 {{\"status\":\"True\"}}，"
        "否则请直接返回 {{\"status\":\"False\"}}。\n\n"
        "问题：{query}\n提供的笔记1：{best_note}\n提供的笔记2：{new_note}\n\n"
        "请根据以上信息进行判断，不要解释，直接返回结果。"
    ),
    "gen_answer": (
        "你是一位专业的中文法律问答助手。现在，你被提供了1个问题，和与问题相关的笔记。"
        "请你结合笔记和你自身的知识回答这些问题。\n"
        "问题：{query}\n\n与问题相关的笔记：{note}\n\n请给出你的回答："
    ),
}


def normalize_completion_url(vllm_url: str) -> str:
    url = vllm_url.rstrip("/")
    if url.endswith("/v1/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/completions"
    return f"{url}/v1/completions"


def load_prompt(prompt_dir: str, name: str) -> str:
    path = os.path.join(prompt_dir, name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return FALLBACK_PROMPTS[name]


def parse_bool_status(text: str) -> bool:
    cleaned = text.strip()
    try:
        return str(json.loads(cleaned).get("status", "")).lower() == "true"
    except Exception:
        match = re.search(r'"?status"?\s*:\s*"?([^"}]+)"?', cleaned, flags=re.I)
        if match:
            return match.group(1).strip().lower() == "true"
        return "true" in cleaned.lower()


def format_refs(refs: List[str]) -> str:
    if not refs:
        return "未找到相关的法律条文。"
    return "\n\n".join(f"法条参考 {idx}:\n{ref}" for idx, ref in enumerate(refs, start=1))


class AdaptiveNoteLawRetriever:
    """Adaptive-Note CRUD-style BGE + FAISS retriever, with a law corpus."""

    def __init__(self, index_path: str, chunk_path: str, embedding_model: str, device: str):
        self.index = faiss.read_index(index_path)
        self.embed_model = SentenceTransformer(embedding_model, device=device)
        with open(chunk_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

    def retrieve(self, query: str, topk: int) -> Tuple[List[str], float]:
        start = time.time()
        feature = self.embed_model.encode([query])
        _, match_id = self.index.search(feature, topk)
        refs = [self.raw_data[i] for i in match_id[0] if 0 <= i < len(self.raw_data)]
        return refs, time.time() - start


class AdaptiveNoteVLLMAgent:
    def __init__(
        self,
        vllm_url: str,
        model_name: str,
        prompt_dir: str,
        retriever: AdaptiveNoteLawRetriever,
        retrieve_top_k: int,
        max_top_k: int,
        max_step: int,
        max_fail_step: int,
        request_timeout: int,
        max_tokens: int,
    ):
        self.vllm_url = normalize_completion_url(vllm_url)
        self.model_name = model_name
        self.retriever = retriever
        self.retrieve_top_k = retrieve_top_k
        self.max_top_k = max_top_k
        self.max_step = max_step
        self.max_fail_step = max_fail_step
        self.request_timeout = request_timeout
        self.max_tokens = max_tokens
        self.prompts = {
            name: load_prompt(prompt_dir, name)
            for name in [
                "first_gen_answer",
                "gen_new_query",
                "refine_note",
                "compare",
                "gen_answer",
            ]
        }

    def complete(self, template_name: str, values: Dict[str, Any]) -> Tuple[str, int, int]:
        prompt = self.prompts[template_name].format(**values)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
        }
        last_error = None
        for attempt in range(3):
            try:
                response = requests.post(
                    self.vllm_url,
                    json=payload,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                data = response.json()
                if "error" in data:
                    raise RuntimeError(data["error"])
                text = data.get("choices", [{}])[0].get("text", "")
                usage = data.get("usage", {})
                return (
                    text.strip(),
                    int(usage.get("prompt_tokens", 0) or 0),
                    int(usage.get("completion_tokens", 0) or 0),
                )
            except Exception as exc:
                last_error = exc
                time.sleep(1 + attempt)
        raise RuntimeError(f"vLLM request failed after retries: {last_error}")

    def run(self, query: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        tool_latency_sec = 0.0
        rag_count = 0
        user_prompt_tokens = 0
        total_prompt_tokens = 0
        total_comp_tokens = 0

        refs, latency = self.retriever.retrieve(query, self.retrieve_top_k)
        tool_latency_sec += latency
        rag_count += 1

        note, p_tok, c_tok = self.complete(
            "first_gen_answer",
            {"query": query, "refs": format_refs(refs)},
        )
        total_prompt_tokens += p_tok
        total_comp_tokens += c_tok
        user_prompt_tokens = p_tok

        best_note = note
        ref_log = [{"refs": refs, "step": 0, "flag": "init_refs"}]
        note_log = [{"note": note, "step": 0, "flag": "init_note"}]
        query_log: List[Dict[str, Any]] = []
        notes_status: List[bool] = []

        for step in range(self.max_step):
            all_refs = {ref for item in ref_log for ref in item["refs"]}
            if len(all_refs) >= self.max_top_k:
                break

            new_query, p_tok, c_tok = self.complete(
                "gen_new_query",
                {
                    "query": query,
                    "note": best_note,
                    "query_log": str(query_log),
                },
            )
            total_prompt_tokens += p_tok
            total_comp_tokens += c_tok

            refs, latency = self.retriever.retrieve((new_query + "\n" + query)[:500], self.retrieve_top_k)
            tool_latency_sec += latency
            rag_count += 1

            if len(all_refs) + self.retrieve_top_k > self.max_top_k:
                max_ref = self.max_top_k - len(all_refs)
                refs = [ref for ref in refs if ref not in all_refs][:max_ref]
            else:
                refs = [ref for ref in refs if ref not in all_refs]

            new_note, p_tok, c_tok = self.complete(
                "refine_note",
                {"query": query, "refs": format_refs(refs), "note": best_note},
            )
            new_note = new_note.replace("\n", "")
            total_prompt_tokens += p_tok
            total_comp_tokens += c_tok

            status_raw, p_tok, c_tok = self.complete(
                "compare",
                {"query": query, "best_note": best_note, "new_note": new_note},
            )
            total_prompt_tokens += p_tok
            total_comp_tokens += c_tok
            status = parse_bool_status(status_raw)
            flag = "True" if status else "False"

            ref_log.append({"refs": refs, "step": step, "flag": flag})
            note_log.append({"note": new_note, "step": step, "flag": flag})
            query_log.append({"query": new_query, "step": step, "flag": flag})

            if status:
                best_note = new_note
            notes_status.append(status)
            if notes_status.count(False) >= self.max_fail_step:
                break

        answer, p_tok, c_tok = self.complete(
            "gen_answer",
            {"query": query, "note": best_note},
        )
        total_prompt_tokens += p_tok
        total_comp_tokens += c_tok

        trace = {
            "note": best_note,
            "query_log": query_log,
            "note_log": note_log,
            "ref_log": ref_log,
        }
        metrics = {
            "tool_latency_sec": tool_latency_sec,
            "rag_count": rag_count,
            "user_prompt_tokens": user_prompt_tokens,
            "main_total_prompt_tokens": total_prompt_tokens,
            "main_total_comp_tokens": total_comp_tokens,
            "final_completion_tokens": c_tok,
        }
        return answer, trace, metrics


def build_output_item(data_item: Dict[str, Any], idx: int, agent: AdaptiveNoteVLLMAgent) -> Dict[str, Any]:
    start_time = time.time()
    instruction = data_item.get("instruction", "")
    question = data_item.get("question", "")
    full_prompt = f"{instruction}\n{question}".strip()

    answer, trace, agent_metrics = agent.run(full_prompt)
    total_time_sec = time.time() - start_time

    total_tokens = (
        agent_metrics["main_total_prompt_tokens"]
        + agent_metrics["main_total_comp_tokens"]
    )
    completion_tokens = agent_metrics.get(
        "final_completion_tokens",
        agent_metrics["main_total_comp_tokens"],
    )
    user_prompt_tokens = agent_metrics.get("user_prompt_tokens", 0)
    inter_agent_tokens = total_tokens - user_prompt_tokens - completion_tokens

    return {
        "origin_idx": idx,
        "instruction": instruction,
        "question": question,
        "prediction": answer,
        "refr": data_item.get("answer", ""),
        "thinking": trace,
        "metrics": {
            "total_time_sec": total_time_sec,
            "tool_latency_sec": agent_metrics["tool_latency_sec"],
            "rag_count": agent_metrics["rag_count"],
            "total_tokens": total_tokens,
            "user_prompt_tokens": user_prompt_tokens,
            "completion_tokens": completion_tokens,
            "inter_agent_tokens": inter_agent_tokens,
        },
    }


def process_file(
    data_path: str,
    output_path: str,
    agent: AdaptiveNoteVLLMAgent,
    max_workers: int,
    limit: int = None,
) -> None:
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if limit:
        dataset = dataset[:limit]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(build_output_item, item, idx, agent): idx
            for idx, item in enumerate(dataset)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Adaptive-Note {os.path.basename(data_path)}",
        ):
            try:
                results.append(future.result())
            except Exception as exc:
                idx = futures[future]
                logger.exception("Adaptive-Note failed on %s item %s", data_path, idx)
                item = dataset[idx]
                results.append(
                    {
                        "origin_idx": idx,
                        "instruction": item.get("instruction", ""),
                        "question": item.get("question", ""),
                        "prediction": f"ERROR: {exc}",
                        "refr": item.get("answer", ""),
                        "thinking": {"error": str(exc)},
                        "metrics": {
                            "total_time_sec": 0,
                            "tool_latency_sec": 0,
                            "rag_count": 0,
                            "total_tokens": 0,
                            "user_prompt_tokens": 0,
                            "completion_tokens": 0,
                            "inter_agent_tokens": 0,
                        },
                    }
                )

    results.sort(key=lambda item: item["origin_idx"])
    final_dict = {}
    for idx, item in enumerate(results):
        item.pop("origin_idx", None)
        final_dict[str(idx)] = item

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, indent=4, ensure_ascii=False)
    logger.info("Saved %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vllm_url", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="autorefine-qwen2.5-7b-law-adaptivenote")
    parser.add_argument("--prompt_dir", type=str, default="")
    parser.add_argument("--law_index_path", type=str, required=True)
    parser.add_argument("--law_chunk_path", type=str, required=True)
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-base-zh-v1.5")
    parser.add_argument("--embedding_device", type=str, default="cpu")
    parser.add_argument("--max_step", type=int, default=3)
    parser.add_argument("--max_fail_step", type=int, default=1)
    parser.add_argument("--topk", "--retrieve_top_k", dest="retrieve_top_k", type=int, default=5)
    parser.add_argument("--max_top_k", type=int, default=15)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--request_timeout", type=int, default=2000)
    parser.add_argument("--max_tokens", type=int, default=1280)
    args = parser.parse_args()

    if not args.prompt_dir:
        adaptive_note_dir = os.environ.get(
            "ADAPTIVE_NOTE_DIR",
            os.path.join(BASE_DIR, "Adaptive-Note"),
        )
        args.prompt_dir = os.path.join(adaptive_note_dir, "prompts", "zh")

    retriever = AdaptiveNoteLawRetriever(
        index_path=args.law_index_path,
        chunk_path=args.law_chunk_path,
        embedding_model=args.embedding_model,
        device=args.embedding_device,
    )
    agent = AdaptiveNoteVLLMAgent(
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        prompt_dir=args.prompt_dir,
        retriever=retriever,
        retrieve_top_k=args.retrieve_top_k,
        max_top_k=args.max_top_k,
        max_step=args.max_step,
        max_fail_step=args.max_fail_step,
        request_timeout=args.request_timeout,
        max_tokens=args.max_tokens,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    json_files = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    logger.info("Found %s LawBench files under %s", len(json_files), args.data_dir)
    for data_file in json_files:
        output_file = os.path.join(args.output_dir, os.path.basename(data_file))
        if os.path.exists(output_file):
            logger.info("Skip existing output: %s", output_file)
            continue
        process_file(
            data_path=data_file,
            output_path=output_file,
            agent=agent,
            max_workers=args.workers,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
