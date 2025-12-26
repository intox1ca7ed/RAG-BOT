from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

                                                                          
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_core.config import Config, DEFAULT_CONFIG
from rag_core.index_store import SklearnVersionMismatch
from rag_core.rag import answer_question
from rag_core.embeddings import EmbeddingMetadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

                                                                                                
                                                                                          
                                         


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
    parser.add_argument(
        "--llm_model",
        default="gpt-4o-mini",
        help="LLM model to use for final answer generation (default/recommended: gpt-4o-mini).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=600,
        help="Max tokens for LLM response (when not using --no_llm).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature (when not using --no_llm).",
    )
    parser.add_argument(
        "--llm_timeout",
        type=int,
        default=60,
        help="LLM request timeout in seconds (if supported by client).",
    )
    parser.add_argument(
        "--grounding",
        choices=["strict", "loose", "none"],
        default="strict",
        help="Grounding mode for LLM answers (strict requires sources, loose is permissive).",
    )
    parser.add_argument(
        "--router_backend",
        choices=["rules", "llm", "auto"],
        default="auto",
        help="Routing backend: auto (default user-facing), rules (deterministic regression), llm (forces LLM router; needs OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--show_sources",
        dest="show_sources",
        action="store_true",
        default=True,
        help="Print retrieved sources after the answer (on by default).",
    )
    parser.add_argument(
        "--no_show_sources",
        dest="show_sources",
        action="store_false",
        help="Do not print retrieved sources after the answer.",
    )
    parser.add_argument(
        "--routing_log",
        default=None,
        help="Optional path to append routing events as JSONL.",
    )
    parser.add_argument(
        "--routing_log_include_rationale",
        action="store_true",
        help="Include LLM router rationale in routing log (off by default).",
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
    except Exception as exc:                
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
            llm_model=args.llm_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            llm_timeout=args.llm_timeout,
            grounding=args.grounding,
            router_backend=args.router_backend,
            routing_log=args.routing_log,
            routing_log_include_rationale=args.routing_log_include_rationale,
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
                llm_model=args.llm_model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                llm_timeout=args.llm_timeout,
                grounding=args.grounding,
                router_backend=args.router_backend,
                routing_log=args.routing_log,
                routing_log_include_rationale=args.routing_log_include_rationale,
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
    print(
        f"Intent: {info.get('intent')} (conf={info.get('intent_confidence'):.2f}, rule={info.get('intent_rule')}, backend={info.get('router_backend','rules')} requested={info.get('router_backend_requested','auto')})"
    )
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
    if args.debug and not args.no_llm:
        print(f"LLM: model={args.llm_model} max_tokens={args.max_tokens} temp={args.temperature}")
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
    if args.debug and info.get("delivery_debug"):
        print("Delivery candidates:")
        for row in info["delivery_debug"][:3]:
            print(f"- {row}")
    if args.backend != "auto" and info.get("backend") and info.get("backend") != args.backend:
        print(f"Note: requested backend {args.backend} but index uses {info.get('backend')}; rebuild if needed.")

    def _safe_print(text: str) -> None:
        try:
            print(text)
        except UnicodeEncodeError:
            enc = sys.stdout.encoding or "utf-8"
            print(text.encode(enc, errors="replace").decode(enc))

    def _brief_sources(res: dict, limit: int = 5) -> list[str]:
        seen = set()
        items: list[str] = []
        for chunk, _score in res.get("used_hits") or []:
            meta = chunk.get("metadata", {})
            doc_id = meta.get("doc_id") or meta.get("chunk_id") or ""
            if doc_id in seen:
                continue
            seen.add(doc_id)
            title = meta.get("title") or doc_id or "unknown"
            items.append(f"{title} ({doc_id})")
            if len(items) >= limit:
                break
        return items

    print("Answer:")
    _safe_print(result["answer_text"])
    if args.show_sources:
        print("Sources:")
        if args.debug:
            if result["sources"]:
                for src in result["sources"]:
                    _safe_print(f"- {src}")
            else:
                print("- none")
        else:
            brief = _brief_sources(result)
            if brief:
                for src in brief:
                    _safe_print(f"- {src}")
            else:
                print("- none")
    if args.show_context and result["context_lines"]:
        print("Context:")
        for line in result["context_lines"]:
            _safe_print(line)

                                     
                                                                                                       
                                                                                                                                      
                                                                                        
                                                                                                           


if __name__ == "__main__":
    main()
