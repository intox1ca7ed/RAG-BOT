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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question against the RAG index.")
    parser.add_argument("--question", required=True, help="Question to ask.")
    parser.add_argument("--topk", type=int, default=DEFAULT_CONFIG.top_k, help="Top K chunks to retrieve.")
    parser.add_argument("--show_context", action="store_true", help="Show retrieved context.")
    parser.add_argument("--no_llm", action="store_true", help="Disable LLM answering.")
    parser.add_argument("--rebuild_on_mismatch", action="store_true", help="Rebuild index if sklearn versions differ.")
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
            )
        else:
            print(str(mismatch))
            sys.exit(1)

    info = result["retrieval_info"]
    print(f"Allowed tags: {_format_allowed_tags(info['allowed_tags'])}")
    print(f"Filtering applied: {info['filtered_applied']} (fallback_used={info['filter_warning']})")
    print(f"Top score: {info['top_score']:.3f}")

    print("Answer:")
    print(result["answer_text"])
    print("Sources:")
    if result["sources"]:
        for src in result["sources"]:
            print(f"- {src}")
    else:
        print("- none")
    if args.show_context and result["context_lines"]:
        print("Context:")
        for line in result["context_lines"]:
            print(line)

# Sanity checks (expected behaviour):
# python scripts/ask.py --question "Do I need visa if I stay 20 days in China?" --show_context --no_llm
#   -> Allowed tags: china, visa; filtered retrieval preferred; answer should mention 30-day visa-free window or handoff if uncertain.
# python scripts/ask.py --question "What's your wechat contact?" --show_context --no_llm
#   -> Allowed tags: none; unfiltered; answer should surface WeChat contact details from contacts/FAQ docs.


if __name__ == "__main__":
    main()
