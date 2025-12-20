from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Ensure repo root is on sys.path for local execution without installation
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_core.config import Config, DEFAULT_CONFIG
from rag_core.index_store import SklearnVersionMismatch
from rag_core.rag import answer_question
from rag_core.embeddings import EmbeddingMetadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# NOTE: Routing/refusal tuning pass: add debug flag, optional index rebuild on sklearn mismatch,
# and richer routing diagnostics printing allowed tags, filters, and scores so answers are
# explainable and consistent across runs.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question against the RAG index.")
    parser.add_argument("--question", required=True, help="Question to ask.")
    parser.add_argument("--topk", type=int, default=DEFAULT_CONFIG.top_k, help="Top K chunks to retrieve.")
    parser.add_argument("--show_context", action="store_true", help="Show retrieved context.")
    parser.add_argument("--no_llm", action="store_true", help="Disable LLM answering.")
    parser.add_argument("--rebuild_on_mismatch", action="store_true", help="Rebuild index if sklearn versions differ.")
    parser.add_argument("--debug", action="store_true", help="Print routing/debug details.")
    parser.add_argument(
        "--backend",
        choices=["auto", "local", "tfidf"],
        default="auto",
        help="Desired embedding backend (uses index backend if it differs).",
    )
    parser.add_argument(
        "--st_model",
        default="all-MiniLM-L6-v2",
        help="Preferred sentence-transformers model (used when backend local/auto and available).",
    )
    parser.add_argument(
        "--storage_root",
        type=Path,
        default=DEFAULT_CONFIG.storage_root,
        help="Path to storage directory (defaults to ./storage)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_CONFIG.manifest_csv,
        help="Path to manifest.csv (used if rebuilding).",
    )
    parser.add_argument(
        "--corpus_root",
        type=Path,
        default=DEFAULT_CONFIG.corpus_root,
        help="Path to corpus root (used if rebuilding).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking of top-N chunks (off by default).",
    )
    parser.add_argument(
        "--rerank_mode",
        choices=["off", "on", "auto"],
        default="off",
        help="Rerank mode: off (default), on, or auto (only when sentence-transformers backend).",
    )
    parser.add_argument(
        "--rerank_top_n",
        type=int,
        default=30,
        help="Number of top embedding-ranked chunks to rerank.",
    )
    parser.add_argument(
        "--rerank_top_k",
        type=int,
        default=None,
        help="Override final top-K after rerank (defaults to retrieval top_k).",
    )
    parser.add_argument(
        "--reranker_model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model for reranking.",
    )
    return parser.parse_args()


def _format_allowed_tags(tags: list[str]) -> str:
    return ", ".join(tags) if tags else "none"


def _rebuild_index(args: argparse.Namespace) -> bool:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_index.py"),
        "--rebuild",
        "--manifest",
        str(args.manifest),
        "--corpus_root",
        str(args.corpus_root),
        "--storage_root",
        str(args.storage_root),
    ]
    logging.info("Rebuilding index due to sklearn version mismatch: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return True
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed to rebuild index: %s", exc)
        return False


def main() -> None:
    args = parse_args()
    cfg = Config(storage_root=args.storage_root, manifest_csv=args.manifest, corpus_root=args.corpus_root)
    cfg.ensure_storage_dirs()
    try:
        result = answer_question(
            question=args.question,
            top_k=args.topk,
            use_llm=not args.no_llm,
            show_context=args.show_context,
            config=cfg,
            rebuild_on_mismatch=args.rebuild_on_mismatch,
            debug=args.debug,
            rerank=args.rerank,
            rerank_mode=args.rerank_mode,
            rerank_top_n=args.rerank_top_n,
            rerank_top_k=args.rerank_top_k,
            reranker_model=args.reranker_model,
        )
    except SklearnVersionMismatch as mismatch:
        if args.rebuild_on_mismatch and _rebuild_index(args):
            result = answer_question(
            question=args.question,
            top_k=args.topk,
            use_llm=not args.no_llm,
            show_context=args.show_context,
            config=cfg,
            rebuild_on_mismatch=False,
            debug=args.debug,
            rerank=args.rerank,
            rerank_mode=args.rerank_mode,
            rerank_top_n=args.rerank_top_n,
            rerank_top_k=args.rerank_top_k,
            reranker_model=args.reranker_model,
        )
        else:
            print(str(mismatch))
            sys.exit(1)
    except RuntimeError as exc:
        if "sentence-transformers" in str(exc).lower():
            print(str(exc))
            sys.exit(1)
        raise

    info = result["retrieval_info"]
    print(f"Backend used: {info.get('backend','unknown')} (model={info.get('model_name')})")
    print(f"Allowed tags: {_format_allowed_tags(info['allowed_tags'])}")
    print(
        f"Filtering applied: {info['filtered_applied']} "
        f"(fallback_used={info.get('fallback_used', False)}; reasons={';'.join(info.get('fallback_reasons', [])) or 'none'})"
    )
    print(f"China default priority: {info.get('prefer_china_default', False)}")
    print(f"Intent: {info.get('intent')} (conf={info.get('intent_confidence'):.2f}, rule={info.get('intent_rule')})")
    print(
        f"Processing lock: {info.get('processing_country') or 'none'}; "
        f"Contact strict: {info.get('contact_strict', False)}"
    )
    print(
        f"Rerank: enabled={info.get('rerank_used', False)} "
        f"(mode={args.rerank_mode}, model={info.get('rerank_model')}, device={info.get('rerank_device')}) "
        f"N={info.get('rerank_top_n')} K={info.get('rerank_top_k')} changed={info.get('rerank_changed', 0)}"
    )
    print(f"Shortlisted docs: {', '.join(info.get('shortlisted_docs', [])) or 'none'}")
    print(f"Top score: {info['top_score']:.3f}")
    if args.debug and info.get("contact_allowed_docs"):
        print("Contact allowlist docs:")
        for doc in info["contact_allowed_docs"]:
            print(f"- {doc}")
    if args.debug and info.get("contact_include_reasons"):
        print("Contact include reasons:")
        for reason in info["contact_include_reasons"]:
            print(f"- {reason}")
    if args.debug and info.get("excluded_docs"):
        print("Excluded docs:")
        for reason in info["excluded_docs"]:
            print(f"- {reason}")
    if args.debug and info.get("rerank_table"):
        print("Rerank sample (top 5):")
        for row in info["rerank_table"]:
            print(f"- {row}")
    if args.backend != "auto" and info.get("backend") and info.get("backend") != args.backend:
        print(f"Note: requested backend {args.backend} but index uses {info.get('backend')}; rebuild if needed.")

    def _safe_print(text: str) -> None:
        try:
            print(text)
        except UnicodeEncodeError:
            enc = sys.stdout.encoding or "utf-8"
            print(text.encode(enc, errors="replace").decode(enc))

    print("Answer:")
    _safe_print(result["answer_text"])
    print("Sources:")
    if result["sources"]:
        for src in result["sources"]:
            _safe_print(f"- {src}")
    else:
        print("- none")
    if args.show_context and result["context_lines"]:
        print("Context:")
        for line in result["context_lines"]:
            _safe_print(line)

# Sanity checks (expected behaviour):
# python scripts/ask.py --question "Do I need visa if I stay 20 days in China?" --show_context --no_llm
#   -> Allowed tags: china, visa; filtered retrieval preferred; answer should mention 30-day visa-free window or handoff if uncertain.
# python scripts/ask.py --question "What's your wechat contact?" --show_context --no_llm
#   -> Allowed tags: none; unfiltered; answer should surface WeChat contact details from contacts/FAQ docs.


if __name__ == "__main__":
    main()
