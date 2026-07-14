import argparse
import json
import os
from typing import List

import faiss
from sentence_transformers import SentenceTransformer


def load_law_corpus(corpus_path: str) -> List[str]:
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            law_name = item.get("law_name", "")
            law_duration = item.get("law_duration", "")
            content = item.get("content", "")
            text = "\n".join(part for part in [law_name, law_duration, content] if part).strip()
            if text:
                docs.append(text)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--embedding_model", default="BAAI/bge-base-zh-v1.5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    chunk_path = os.path.join(args.output_dir, "chunk.json")
    index_path = os.path.join(args.output_dir, "law.index")

    docs = load_law_corpus(args.corpus_path)
    if not docs:
        raise ValueError(f"No usable documents loaded from {args.corpus_path}")

    print(f"[INFO] Loaded {len(docs)} law documents from {args.corpus_path}")
    print(f"[INFO] Encoding with {args.embedding_model} on {args.device}")
    model = SentenceTransformer(args.embedding_model, device=args.device)
    embeddings = model.encode(docs, batch_size=args.batch_size, show_progress_bar=True)

    print(f"[INFO] Building FAISS IndexFlatIP: {index_path}")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)

    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    print(f"[INFO] Saved chunks: {chunk_path}")
    print(f"[INFO] Saved index: {index_path}")


if __name__ == "__main__":
    main()
