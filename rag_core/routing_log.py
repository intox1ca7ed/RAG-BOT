from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_routing_event(event: dict[str, Any], path: str) -> None:
    """
    Append a routing event as one JSON line.
    Fields expected in event:
      timestamp, question, router_backend, selected_backend_used, intent,
      allowed_tags, confidence, rule, rationale (optional), shortlisted_docs,
      top_doc_score, top2_doc_score.
    """
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
