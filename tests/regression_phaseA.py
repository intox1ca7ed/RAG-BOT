from __future__ import annotations

import re
import sys
from pathlib import Path

from rag_core.config import Config
from rag_core.rag import answer_question

ROOT = Path(__file__).resolve().parent.parent

QUERIES = [
    "I live in another city. How can I send my passport to you?",
    "I need to get the apostille urgently, how much does it cost and when it's going to be ready?",
    "How much does the furniture tour cost?",
    "How can I contact you? What's your phone contact number?",
]


def _run_query(q: str) -> dict:
    cfg = Config()
    return answer_question(
        question=q,
        top_k=6,
        use_llm=False,
        show_context=False,
        config=cfg,
        rebuild_on_mismatch=False,
        debug=False,
    )


def main() -> int:
    print("Running Phase A regression on 4 queries...")
    results = []
    for idx, q in enumerate(QUERIES, 1):
        print(f"\nQuery {idx}: {q}")
        res = _run_query(q)
        results.append(res)
        for chunk, score in res["hits"][:3]:
            meta = chunk.get("metadata", {})
            print(f"- {meta.get('doc_id')} score={score:.3f} text={chunk.get('text','')[:120].replace(chr(10),' ')}")

    # Assertions
    send_res = results[0]
    send_texts = [c.get("text", "").lower() for c, _ in send_res["hits"]]
    if not any("courier" in t or "flying envelope" in t for t in send_texts):
        print("Regression failed: Query 1 missing courier/flying envelope chunk.")
        return 1

    apo_res = results[1]
    apo_texts = [c.get("text", "").lower() for c, _ in apo_res["hits"]]
    if not any("apostille" in t and "day" in t for t in apo_texts):
        print("Regression failed: Query 2 missing apostille days chunk.")
        return 1

    furn_res = results[2]
    furn_doc_ids = [c.get("metadata", {}).get("doc_id", "").lower() for c, _ in furn_res["hits"]]
    if any("duties" in d for d in furn_doc_ids):
        print("Regression failed: Query 3 retrieved duties doc for furniture tour cost.")
        return 1

    contact_res = results[3]
    contact_docs = [c.get("metadata", {}).get("doc_id", "").lower() for c, _ in contact_res["hits"]]
    if not any("contacts" in d for d in contact_docs):
        print("Regression failed: Query 4 missing contacts chunk.")
        return 1

    print("Phase A regression passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
