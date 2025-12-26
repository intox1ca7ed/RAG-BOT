from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

                          
REPO_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_core.config import Config, DEFAULT_CONFIG
from rag_core.rag import answer_question


INTENT_ALIASES = {
    "apostille_vs_legalization": {"apostille_vs_legalization", "apostille"},
    "cargo_delivery": {"cargo_delivery", "delivery"},
}


@dataclass
class CaseResult:
    case_id: str
    track: str
    status: str
    intent_expected: str
    intent_predicted: str
    intent_ok: bool
    retrieval_ok: bool
    content_ok: bool
    missing_includes: list[str]
    forbidden_hits: list[str]
    top_docs: list[str]
    router_backend_used: str
    confidence: float
    answer_text: str
    question: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate answers against a gold set.")
    parser.add_argument(
        "--cases_file",
        default=str(Path("tests") / "answer_gold.json"),
        help="Path to answer gold JSON file.",
    )
    parser.add_argument(
        "--track",
        choices=["baseline", "demo", "both"],
        default="both",
        help="Evaluation track: baseline (rules + no_llm), demo (auto + llm when allowed), or both.",
    )
    parser.add_argument(
        "--report_json",
        default=str(Path("logs") / "eval_report.json"),
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--report_txt",
        default=str(Path("logs") / "eval_report.txt"),
        help="Path to write text report.",
    )
    parser.add_argument(
        "--top_docs",
        type=int,
        default=5,
        help="Number of top doc_ids to record.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of cases (for smoke tests).",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("answer_gold.json must be a JSON array")
    return data


def _match_pattern(text: str, pattern: str) -> bool:
    if pattern.startswith("re:"):
        return re.search(pattern[3:], text, flags=re.IGNORECASE) is not None
    return pattern.lower() in text.lower()


def _intent_matches(expected: str, actual: str) -> bool:
    aliases = INTENT_ALIASES.get(expected, {expected})
    return actual in aliases


def _collect_top_docs(result: dict, limit: int) -> list[str]:
    doc_ids: list[str] = []
    for chunk, _score in result.get("used_hits") or []:
        meta = chunk.get("metadata", {})
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
        if len(doc_ids) >= limit:
            break
    return doc_ids


def _evaluate_case(
    case: dict,
    track: str,
    cfg: Config,
    top_docs: int,
) -> CaseResult:
    question = case["question"]
    expected_intent = case["expected_intent"]
    must_include = case.get("must_include", [])
    must_not_include = case.get("must_not_include", [])
    expected_docs_any = case.get("expected_docs_any", [])

    api_key = os.getenv("OPENAI_API_KEY")
    use_llm = False
    if track == "demo":
        use_llm = case.get("mode") == "llm"
        if use_llm and not api_key:
            return CaseResult(
                case_id=case["id"],
                track=track,
                status="skipped",
                intent_expected=expected_intent,
                intent_predicted="",
                intent_ok=False,
                retrieval_ok=False,
                content_ok=False,
                missing_includes=[],
                forbidden_hits=[],
                top_docs=[],
                router_backend_used="",
                confidence=0.0,
                answer_text="",
                question=question,
            )

    if track == "baseline":
        use_llm = False

    router_backend = "rules" if track == "baseline" else "auto"
    try:
        result = answer_question(
            question=question,
            use_llm=use_llm,
            show_context=False,
            config=cfg,
            rebuild_on_mismatch=False,
            debug=False,
            router_backend=router_backend,
        )
    except RuntimeError as exc:
        if "sentence-transformers" in str(exc).lower():
            return CaseResult(
                case_id=case["id"],
                track=track,
                status="skipped",
                intent_expected=expected_intent,
                intent_predicted="",
                intent_ok=False,
                retrieval_ok=False,
                content_ok=False,
                missing_includes=[],
                forbidden_hits=[],
                top_docs=[],
                router_backend_used="",
                confidence=0.0,
                answer_text="",
                question=question,
            )
        raise

    info = result["retrieval_info"]
    predicted_intent = info.get("intent", "none")
    intent_ok = _intent_matches(expected_intent, predicted_intent)

    top_doc_ids = _collect_top_docs(result, top_docs)
    retrieval_ok = True
    if expected_docs_any:
        retrieval_ok = any(doc_id in top_doc_ids for doc_id in expected_docs_any)

    answer_text = result.get("answer_text", "")
    missing = [pat for pat in must_include if not _match_pattern(answer_text, pat)]
    forbidden = [pat for pat in must_not_include if _match_pattern(answer_text, pat)]
    content_ok = not missing and not forbidden

    status = "passed" if (intent_ok and retrieval_ok and content_ok) else "failed"
    return CaseResult(
        case_id=case["id"],
        track=track,
        status=status,
        intent_expected=expected_intent,
        intent_predicted=predicted_intent,
        intent_ok=intent_ok,
        retrieval_ok=retrieval_ok,
        content_ok=content_ok,
        missing_includes=missing,
        forbidden_hits=forbidden,
        top_docs=top_doc_ids,
        router_backend_used=info.get("router_backend", ""),
        confidence=float(info.get("intent_confidence", 0.0) or 0.0),
        answer_text=answer_text,
        question=question,
    )


def _summarize(results: list[CaseResult]) -> dict[str, Any]:
    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r.status == "passed"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
    }
    by_intent: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0})
    for r in results:
        bucket = by_intent[r.intent_expected]
        bucket[r.status] += 1
    summary["by_intent"] = dict(by_intent)
    return summary


def run_eval(
    cases_file: Path,
    track: str,
    report_json: Path,
    report_txt: Path,
    top_docs: int = 5,
    limit: int | None = None,
) -> dict[str, Any]:
    cases = load_cases(cases_file)
    if limit is not None:
        cases = cases[:limit]
    cfg = Config()

    tracks = [track] if track != "both" else ["baseline", "demo"]
    all_results: dict[str, list[CaseResult]] = {}
    for tr in tracks:
        results = [_evaluate_case(case, tr, cfg, top_docs) for case in cases]
        all_results[tr] = results

    report = {
        "tracks": {},
        "cases": {},
    }
    for tr, results in all_results.items():
        report["tracks"][tr] = _summarize(results)
        for res in results:
            report["cases"].setdefault(res.case_id, {})
            report["cases"][res.case_id][tr] = res.__dict__

    if track == "both":
        baseline = {r.case_id: r for r in all_results["baseline"]}
        demo = {r.case_id: r for r in all_results["demo"]}
        report["diff"] = {
            "baseline_fail_demo_pass": [
                cid
                for cid, b in baseline.items()
                if b.status == "failed" and demo.get(cid) and demo[cid].status == "passed"
            ],
            "baseline_pass_demo_fail": [
                cid
                for cid, b in baseline.items()
                if b.status == "passed" and demo.get(cid) and demo[cid].status == "failed"
            ],
        }

    report_json_path = report_json
    report_txt_path = report_txt
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_txt_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines: list[str] = []
    for tr, summary in report["tracks"].items():
        lines.append(f"Track: {tr}")
        lines.append(
            f"  total={summary['total']} passed={summary['passed']} failed={summary['failed']} skipped={summary['skipped']}"
        )
        lines.append("")
    lines.append("Failed cases:")
    for tr, results in all_results.items():
        for res in results:
            if res.status != "failed":
                continue
            snippet = res.answer_text.replace("\n", " ")[:240]
            lines.append(f"- [{tr}] {res.case_id}: {res.question}")
            lines.append(f"  intent expected={res.intent_expected} got={res.intent_predicted}")
            lines.append(f"  top_docs={', '.join(res.top_docs) or 'none'}")
            lines.append(f"  missing={res.missing_includes} forbidden={res.forbidden_hits}")
            lines.append(f"  answer_snippet={snippet}")
    if track == "both":
        lines.append("")
        lines.append("Diff:")
        lines.append(f"  baseline_fail_demo_pass={report.get('diff', {}).get('baseline_fail_demo_pass', [])}")
        lines.append(f"  baseline_pass_demo_fail={report.get('diff', {}).get('baseline_pass_demo_fail', [])}")

    report_txt_path.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = run_eval(
        cases_file=Path(args.cases_file),
        track=args.track,
        report_json=Path(args.report_json),
        report_txt=Path(args.report_txt),
        top_docs=args.top_docs,
        limit=args.limit,
    )
    for tr, summary in report["tracks"].items():
        print(
            f"{tr}: total={summary['total']} passed={summary['passed']} failed={summary['failed']} skipped={summary['skipped']}"
        )
        by_intent = summary.get("by_intent", {})
        if by_intent:
            for intent, counts in by_intent.items():
                print(
                    f"  {intent}: passed={counts.get('passed',0)} failed={counts.get('failed',0)} skipped={counts.get('skipped',0)}"
                )
    print(f"Report JSON: {args.report_json}")
    print(f"Report TXT: {args.report_txt}")


if __name__ == "__main__":
    main()
