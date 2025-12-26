from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

                          
REPO_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_core.config import Config, DEFAULT_CONFIG
from rag_core.index_store import IndexStore, SklearnVersionMismatch
from rag_core.retrieval import retrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate router modes (rules/auto/llm) without final LLM.")
    parser.add_argument("--questions_file", required=True, help="Text file with one question per line.")
    parser.add_argument("--topk", type=int, default=DEFAULT_CONFIG.top_k, help="Top K chunks to retrieve.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs for routing.")
    parser.add_argument(
        "--llm_model",
        default="gpt-4o-mini",
        help="LLM model to use for LLM router mode.",
    )
    parser.add_argument(
        "--llm_timeout",
        type=int,
        default=60,
        help="LLM request timeout for router calls (if used).",
    )
    parser.add_argument(
        "--storage_root",
        type=Path,
        default=DEFAULT_CONFIG.storage_root,
        help="Path to storage directory (defaults to ./storage)",
    )
    return parser.parse_args()


def load_questions(path: Path) -> list[str]:
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        q = line.strip()
        if q:
            lines.append(q)
    return lines


def main() -> None:
    args = parse_args()
    cfg = Config(storage_root=args.storage_root)
    cfg.ensure_storage_dirs()
    try:
        index_store, embed_model, _ = IndexStore.load(cfg.index_dir, rebuild_on_mismatch=False)
    except SklearnVersionMismatch as exc:
        logging.error(str(exc))
        return

    questions = load_questions(Path(args.questions_file))
    modes = ["rules", "auto", "llm"]
    has_key = bool(os.getenv("OPENAI_API_KEY"))

    print("question\tmode\tselected_backend\tintent\tconfidence\tallowed_tags\ttop_docs")
    for q in questions:
        for mode in modes:
            if mode == "llm" and not has_key:
                print(f"{q}\t{mode}\tSKIP(no_key)\t-\t-\t-\t-")
                continue
            info = {}
            hits = []
            try:
                hits, _confidence, info = retrieve(
                    q,
                    index_store,
                    embed_model,
                    top_k=args.topk,
                    config=cfg,
                    debug=args.debug,
                    router_backend=mode,
                    llm_model=args.llm_model,
                    llm_timeout=args.llm_timeout,
                )
            except Exception as exc:                
                print(f"{q}\t{mode}\tERROR({exc})\t-\t-\t-\t-")
                continue
            backend_used = info.get("router_backend", mode)
            top_docs = ",".join(info.get("shortlisted_docs", [])[:3])
            intent = info.get("intent")
            conf = info.get("intent_confidence")
            allowed_tags = ",".join(info.get("allowed_tags", []))
            print(f"{q}\t{mode}\t{backend_used}\t{intent}\t{conf:.2f}\t{allowed_tags}\t{top_docs}")


if __name__ == "__main__":
    main()
