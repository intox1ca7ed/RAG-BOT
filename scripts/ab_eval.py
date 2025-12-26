import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rag_core.config import Config
import eval_answers as eval_mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A/B eval with rules vs LLM routing.")
    parser.add_argument("--gold", default=str(Path("tests") / "answer_gold.json"))
    parser.add_argument("--index_dir", default=str(Path("storage") / "index"))
    parser.add_argument("--chunks", default=str(Path("storage") / "chunks" / "chunks.jsonl"))
    parser.add_argument("--out_dir", default=str(Path("artifacts") / "ab_eval"))
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def is_abstain(text: str) -> bool:
    t = (text or "").lower()
    patterns = [
        "i can't find this in the provided sources",
        "please contact the office",
        "please reach out",
        "insufficient information",
        "i couldn't find",
        "please call the main office",
    ]
    return any(p in t for p in patterns)


def failure_reasons(res) -> list[str]:
    if res.status == "passed":
        return []
    reasons = []
    if not res.intent_ok:
        reasons.append("intent_mismatch")
    if not res.retrieval_ok:
        reasons.append("retrieval_miss")
    for item in res.missing_includes:
        reasons.append(f"missing:{item}")
    for item in res.forbidden_hits:
        reasons.append(f"forbidden:{item}")
    if res.status == "skipped":
        reasons.append("skipped")
    return reasons


def can_run_llm_router() -> tuple[bool, str]:
    if not os.getenv("OPENAI_API_KEY"):
        return False, "skipped_missing_api_key"
    try:
        import openai
    except Exception:
        return False, "skipped_missing_openai"
    return True, ""


def run_variant(variant: str, router_backend: str, cfg: Config, cases: list[dict], top_k: int, top_docs: int):
    if router_backend == "llm":
        ok, reason = can_run_llm_router()
        if not ok:
            details = []
            for case in cases:
                details.append(
                    {
                        "question_id": case.get("id"),
                        "variant": variant,
                        "status": reason,
                        "passed": False,
                        "failure_reasons": [reason],
                        "used_sources": [],
                        "router_intent": "",
                        "router_tags": [],
                        "answer_preview": "",
                    }
                )
            summary = {
                "variant": variant,
                "total": len(cases),
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "abstain_count": 0,
                "abstain_rate": 0.0,
                "status": reason,
            }
            return summary, details

    results = []
    total_cases = len(cases)
    for idx, case in enumerate(cases, start=1):
        res = eval_mod._evaluate_case(
            case=case,
            track=variant,
            cfg=cfg,
            top_docs=top_docs,
            router_backend_override=router_backend,
            use_llm_override=False,
            top_k_override=top_k,
        )
        results.append(res)
        if total_cases:
            width = 24
            filled = int(width * idx / total_cases)
            bar = "#" * filled + "-" * (width - filled)
            print(f"{variant}: [{bar}] {idx}/{total_cases}", end="\r", flush=True)
    if total_cases:
        print()

    total = len(results)
    passed = sum(1 for r in results if r.status == "passed")
    failed = sum(1 for r in results if r.status == "failed")
    pass_rate = (passed / total) if total else 0.0
    abstain_count = sum(1 for r in results if is_abstain(r.answer_text))
    abstain_rate = (abstain_count / total) if total else 0.0

    details = []
    for res in results:
        details.append(
            {
                "question_id": res.case_id,
                "variant": variant,
                "status": res.status,
                "passed": res.status == "passed",
                "failure_reasons": failure_reasons(res),
                "used_sources": res.top_docs,
                "router_intent": res.intent_predicted,
                "router_tags": res.router_tags,
                "answer_preview": (res.answer_text or "").replace("\n", " ")[:180],
            }
        )

    summary = {
        "variant": variant,
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(pass_rate, 4),
        "abstain_count": abstain_count,
        "abstain_rate": round(abstain_rate, 4),
        "status": "ok",
    }
    return summary, details


def write_summary_csv(path: Path, summaries: list[dict]) -> None:
    fieldnames = ["variant", "total", "passed", "failed", "pass_rate", "abstain_count", "abstain_rate"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k, "") for k in fieldnames})


def write_summary_txt(path: Path, summaries: list[dict]) -> None:
    lines = [
        "Variant A uses rules-based routing only.",
        "Variant B uses LLM-based routing only.",
    ]
    for s in summaries:
        if s.get("status") != "ok":
            lines.append(
                f"Variant {s.get('variant')}: {s.get('status')} (total={s.get('total')})."
            )
            continue
        lines.append(
            f"Variant {s.get('variant')}: total={s.get('total')} passed={s.get('passed')} failed={s.get('failed')} pass_rate={s.get('pass_rate'):.2f} abstain_rate={s.get('abstain_rate'):.2f}."
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_pass_rates(path: Path, summaries: list[dict]) -> None:
    variants = [s.get("variant") for s in summaries]
    pass_rates = [s.get("pass_rate", 0.0) for s in summaries]
    abstain_rates = [s.get("abstain_rate", 0.0) for s in summaries]

    x = np.arange(len(variants))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, pass_rates, width, label="pass_rate")
    plt.bar(x + width / 2, abstain_rates, width, label="abstain_rate")
    plt.xticks(x, variants)
    plt.ylabel("Rate")
    plt.title("A/B pass and abstain rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = eval_mod.load_cases(Path(args.gold))
    cfg = Config()

    summaries = []
    details = []

    summary_a, details_a = run_variant("A", "rules", cfg, cases, top_k=args.k, top_docs=5)
    summaries.append(summary_a)
    details.extend(details_a)

    summary_b, details_b = run_variant("B", "llm", cfg, cases, top_k=args.k, top_docs=5)
    summaries.append(summary_b)
    details.extend(details_b)

    write_summary_csv(out_dir / "ab_summary.csv", summaries)
    (out_dir / "ab_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    write_summary_txt(out_dir / "ab_summary.txt", summaries)
    plot_pass_rates(out_dir / "ab_pass_rate.png", summaries)

    print(f"Wrote {out_dir / 'ab_summary.csv'}")
    print(f"Wrote {out_dir / 'ab_details.json'}")
    print(f"Wrote {out_dir / 'ab_pass_rate.png'}")
    print(f"Wrote {out_dir / 'ab_summary.txt'}")


if __name__ == "__main__":
    main()
