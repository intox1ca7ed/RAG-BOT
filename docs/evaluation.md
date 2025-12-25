# Evaluation Harness

Baseline (no API key required):

```
python scripts/eval_answers.py --track baseline
```

Demo evaluation (uses LLM when available):

```
set OPENAI_API_KEY=your_key_here
python scripts/eval_answers.py --track demo
```

Both tracks with comparison:

```
python scripts/eval_answers.py --track both
```

Reports:
- `logs/eval_report.json` contains full per-case results.
- `logs/eval_report.txt` lists failed cases with missing/forbidden content and a short answer snippet.

Add new test cases:
1) Append to `tests/answer_gold.json` with a unique `id`, question, expected intent, `must_include` strings (or `re:` regex), `must_not_include`, and `expected_docs_any`.
2) Keep `must_include` minimal (1–3 facts) to avoid brittleness.
