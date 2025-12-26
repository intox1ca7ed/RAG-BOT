from __future__ import annotations

import os
from pathlib import Path

                          
REPO_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_core.config import Config
from rag_core.rag import answer_question


QUESTIONS = [
    "What is your WeChat id?",
    "What are your opening hours?",
    "How can I send my passport if I can't come in person?",
    "Do I need a visa for 20 days in China?",
    "How much for Q2 family visa to China?",
    "How long does China visa processing take?",
    "How much does the consultant/translator cost on the furniture tour?",
    "What is the capacity of a 20-foot container?",
    "What is the export clearance price in furniture tour services?",
    "What is the minimum cargo delivery and how long does it take?",
]


def _top_doc_ids(result: dict, limit: int = 3) -> list[str]:
    doc_ids: list[str] = []
    for chunk, _score in result.get("used_hits") or []:
        meta = chunk.get("metadata", {})
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
        if len(doc_ids) >= limit:
            break
    return doc_ids


def main() -> None:
    use_llm = bool(os.getenv("OPENAI_API_KEY"))
    cfg = Config()
    for q in QUESTIONS:
        result = answer_question(
            question=q,
            use_llm=use_llm,
            show_context=False,
            config=cfg,
            rebuild_on_mismatch=False,
            debug=False,
            router_backend="auto",
        )
        info = result["retrieval_info"]
        print("=" * 72)
        print(f"Question: {q}")
        print(
            f"Router: backend={info.get('router_backend')} intent={info.get('intent')} tags={','.join(info.get('allowed_tags', [])) or 'none'}"
        )
        print("Answer:")
        print(result["answer_text"])
        sources = _top_doc_ids(result)
        if sources:
            print(f"Sources: {', '.join(sources)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
