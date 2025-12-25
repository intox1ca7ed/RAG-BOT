from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

# Ensure repo root is on sys.path for local execution without installation
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_core.chunking import chunk_corpus
from rag_core.config import Config, DEFAULT_CONFIG
from rag_core.embeddings import EmbeddingModel
from rag_core.index_store import IndexStore
from rag_core.loaders import load_corpus
from rag_core.manifest import load_manifest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_index")

# NOTE: Embeddings tuning: supports backend auto/local/tfidf with optional sentence-transformers model.
# Writes metadata used for downstream routing/refusal/debug; chunk ordering matches embeddings.npy/vectors.npy.

def write_chunks_jsonl(chunks: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAG index from manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CONFIG.manifest_csv)
    parser.add_argument("--corpus_root", type=Path, default=DEFAULT_CONFIG.corpus_root)
    parser.add_argument("--storage_root", type=Path, default=DEFAULT_CONFIG.storage_root)
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if index exists.")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CONFIG.chunk_size)
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CONFIG.chunk_overlap)
    parser.add_argument(
        "--backend",
        choices=["auto", "local", "tfidf"],
        default="auto",
        help="Embedding backend: auto prefers sentence-transformers if installed.",
    )
    parser.add_argument(
        "--st_model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (when backend local/auto and available).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        manifest_csv=args.manifest,
        manifest_json=args.manifest.with_suffix(".json"),
        corpus_root=args.corpus_root,
        storage_root=args.storage_root,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=DEFAULT_CONFIG.top_k,
        confidence_threshold=DEFAULT_CONFIG.confidence_threshold,
    )
    cfg.ensure_storage_dirs()

    index_dir = cfg.index_dir
    if not args.rebuild and index_dir.exists() and any(index_dir.iterdir()):
        logger.info("Index already exists at %s. Use --rebuild to overwrite.", index_dir)
        return

    logger.info("Loading manifest from %s", cfg.manifest_csv)
    records = load_manifest(cfg)
    # Post-process topics to disambiguate furniture tour vs customs duties
    renamed = 0
    for rec in records:
        title = str(rec.get("title", "")).lower()
        local_path = str(rec.get("local_path", "")).lower()
        doc_id = str(rec.get("doc_id", "")).lower()
        duties_hit = ("furniture" in title and "duties" in title) or ("furniture" in local_path and "duties" in local_path)
        if duties_hit:
            if rec.get("topic") != "customs_duties":
                rec["topic"] = "customs_duties"
                renamed += 1
            tags = str(rec.get("tags", "")).split(",")
            for tag in ("duties", "customs"):
                if tag not in tags:
                    tags.append(tag)
            rec["tags"] = ",".join(t.strip() for t in tags if t.strip())
        if "furniture tour" in title or "furniture_tour" in doc_id:
            if rec.get("topic") != "furniture_tour":
                rec["topic"] = "furniture_tour"
                renamed += 1
    if renamed:
        logger.warning("Updated topics/tags for %d manifest records (furniture/customs disambiguation).", renamed)
    docs = load_corpus(records, cfg)
    logger.info("Loaded %d documents", len(docs))

    logger.info("Chunking documents (size=%s overlap=%s)", cfg.chunk_size, cfg.chunk_overlap)
    chunks = chunk_corpus(docs, cfg.chunk_size, cfg.chunk_overlap)
    logger.info("Created %d chunks", len(chunks))

    logger.info("Embedding chunks")
    prefer = None
    if args.backend == "local":
        prefer = "sentence-transformers"
    elif args.backend == "tfidf":
        prefer = "tfidf"
    embed_model = EmbeddingModel(prefer=prefer, model_name=args.st_model)
    vectors = embed_model.fit([c["text"] for c in chunks])
    if vectors.size == 0:
        raise SystemExit("No vectors produced; check input corpus.")

    vectorizer_path = None
    if embed_model.backend == "tfidf":
        vectorizer_path = "vectorizer.pkl"
        embed_model.save_vectorizer(index_dir / vectorizer_path)
    elif embed_model.backend == "simple":
        vectorizer_path = "simple_vocab.json"
        embed_model.save_vectorizer(index_dir / vectorizer_path)

    meta_vectorizer_path = index_dir / vectorizer_path if vectorizer_path else None
    index_store = IndexStore.build(
        vectors,
        chunks,
        embed_model.to_metadata(vectorizer_path=meta_vectorizer_path),
        meta_info={
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "backend_requested": args.backend,
            "st_model": args.st_model,
        },
    )
    logger.info("Saving index to %s", index_dir)
    index_store.save(index_dir)

    write_chunks_jsonl(chunks, cfg.chunks_path)
    logger.info("Done. Chunks saved to %s", cfg.chunks_path)


if __name__ == "__main__":
    main()
