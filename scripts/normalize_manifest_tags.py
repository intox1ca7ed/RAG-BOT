from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_core.config import DEFAULT_CONFIG
from rag_core.manifest import load_csv, load_json

COUNTRY_TAGS = ["china", "japan", "taiwan"]
TOPIC_TAGS = ["visa", "contact", "form", "legalization", "consulate", "furniture", "cargo"]
SECTION_TAGS = ["prices", "hours", "locations", "documents", "faq", "general"]

CANONICAL_ORDER = COUNTRY_TAGS + TOPIC_TAGS + SECTION_TAGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize manifest tags into a controlled vocabulary.")
    parser.add_argument("--manifest_csv", type=Path, default=DEFAULT_CONFIG.manifest_csv)
    parser.add_argument("--manifest_json", type=Path, default=DEFAULT_CONFIG.manifest_json)
    parser.add_argument("--write_in_place", action="store_true", help="Overwrite manifest.csv/json instead of writing cleaned copies.")
    return parser.parse_args()


def _tokenize_tags(raw_tags: str | list | None) -> list[str]:
    if raw_tags is None:
        return []
    if isinstance(raw_tags, list):
        items = raw_tags
    else:
        items = str(raw_tags).split(",")
    cleaned = []
    for item in items:
        t = item.strip().lower().replace("-", "_")
        if not t:
            continue
        t = t.replace(" ", "_")
        cleaned.append(t)
    return cleaned


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    low = text.lower()
    return any(kw in low for kw in keywords)


def _map_to_canonical(raw_tags: list[str], title: str, local_path: str) -> set[str]:
    canonical = set()
    all_fields = raw_tags + [title.lower(), local_path.lower()]

    def add_if(keyword_list: list[str], tag: str) -> None:
        if any(kw in field for kw in keyword_list for field in all_fields):
            canonical.add(tag)

    add_if(["china", "prc"], "china")
    add_if(["japan"], "japan")
    add_if(["taiwan"], "taiwan")
    add_if(["visa", "entry", "visa_free", "visa-free"], "visa")
    add_if(["contact", "contacts", "phone", "telephone", "whatsapp", "email", "mail"], "contact")
    add_if(["form", "questionnaire"], "form")
    add_if(["legal", "legalization", "notary", "apostille"], "legalization")
    add_if(["consulate", "embassy"], "consulate")
    add_if(["furniture", "tour"], "furniture")
    add_if(["cargo", "customs", "delivery", "container"], "cargo")
    add_if(["price", "cost", "fee"], "prices")
    add_if(["hour", "schedule", "working"], "hours")
    add_if(["where", "places", "location", "cities"], "locations")
    add_if(["document", "docs", "requirements"], "documents")
    add_if(["faq", "q&a", "questions"], "faq")
    return canonical


def _sort_tags(tags: set[str]) -> list[str]:
    order = {tag: i for i, tag in enumerate(CANONICAL_ORDER)}
    return sorted(tags, key=lambda t: order.get(t, 999))


def _derive_structured(tags: list[str]) -> tuple[str, str, str]:
    tag_set = set(tags)
    countries = [t for t in COUNTRY_TAGS if t in tag_set]
    if len(countries) > 1:
        country = "multi"
    elif countries:
        country = countries[0]
    else:
        country = "none"

    topic_priority = ["contact", "visa", "legalization", "consulate", "form", "furniture", "cargo", "general"]
    section_priority = ["prices", "hours", "locations", "documents", "faq", "general"]

    topic = next((t for t in topic_priority if t in tag_set), "general")
    section = next((s for s in section_priority if s in tag_set), "general")
    return country, topic, section


def normalize_record(record: dict) -> tuple[dict, bool, set[str]]:
    original_tags = _tokenize_tags(record.get("tags", ""))
    title = str(record.get("title", ""))
    local_path = str(record.get("local_path", ""))

    canonical = _map_to_canonical(original_tags, title, local_path)
    if not canonical:
        canonical.add("general")

    sorted_tags = _sort_tags(canonical)
    country, topic, section = _derive_structured(sorted_tags)

    changed = set(original_tags) != set(sorted_tags)
    new_record = dict(record)
    new_record["tags"] = ",".join(sorted_tags)
    new_record["country"] = country
    new_record["topic"] = topic
    new_record["section"] = section
    return new_record, changed, set(original_tags)


def load_manifest_records(manifest_csv: Path, manifest_json: Path) -> list[dict]:
    if manifest_csv.exists():
        return load_csv(manifest_csv)
    if manifest_json.exists():
        return load_json(manifest_json)
    raise FileNotFoundError("No manifest.csv or manifest.json found.")


def write_outputs(records: list[dict], path_csv: Path, path_json: Path) -> None:
    if records:
        fieldnames = list(records[0].keys())
    else:
        fieldnames = []
    with path_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with path_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def summarize(before_counts: collections.Counter, after_counts: collections.Counter, changed: int, total: int, multi_country_ids: list[str]) -> None:
    print(f"Total records processed: {total}")
    print(f"Records with changed tags: {changed}")
    print("Top 20 tag counts before:")
    for tag, cnt in before_counts.most_common(20):
        print(f"  {tag}: {cnt}")
    print("Top 20 tag counts after:")
    for tag, cnt in after_counts.most_common(20):
        print(f"  {tag}: {cnt}")
    if multi_country_ids:
        print("Records with multiple countries detected (set country=multi):")
        for doc_id in multi_country_ids:
            print(f"  {doc_id}")


def main() -> None:
    args = parse_args()
    records = load_manifest_records(args.manifest_csv, args.manifest_json)

    cleaned_records = []
    changed_count = 0
    before_counter: collections.Counter[str] = collections.Counter()
    after_counter: collections.Counter[str] = collections.Counter()
    multi_country_ids: list[str] = []

    for rec in records:
        cleaned, changed, original_set = normalize_record(rec)
        cleaned_records.append(cleaned)
        changed_count += int(changed)
        before_counter.update(original_set)
        after_counter.update(cleaned["tags"].split(",") if cleaned.get("tags") else [])
        if cleaned.get("country") == "multi":
            multi_country_ids.append(cleaned.get("doc_id", ""))

    out_csv = args.manifest_csv if args.write_in_place else args.manifest_csv.with_name("manifest.cleaned.csv")
    out_json = args.manifest_json if args.write_in_place else args.manifest_json.with_name("manifest.cleaned.json")

    write_outputs(cleaned_records, out_csv, out_json)
    summarize(before_counter, after_counter, changed_count, len(records), multi_country_ids)


if __name__ == "__main__":
    main()
