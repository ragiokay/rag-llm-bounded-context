# migrate_payloads.py
#
# Adds new payload fields to existing Qdrant points WITHOUT re-embedding.
#
# Two modes:
#   --auto   : auto-parse maven_ere_causal (extracts cause/effect from policy field)
#   --manual : read manual_updates.json and apply updates to any collection
#
# Usage:
#   python embedding/migrate_payloads.py --auto
#   python embedding/migrate_payloads.py --manual embedding/manual_updates.json
#   python embedding/migrate_payloads.py --auto --manual embedding/manual_updates.json

import os
import sys
import json
import argparse
import re
sys.path.insert(0, os.path.dirname(__file__))

from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_PREFIX = os.getenv("COLLECTION_PREFIX", "")


def _get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=300)


# ---------------------------------------------------------------------------
# Auto mode: maven_ere_causal
# ---------------------------------------------------------------------------

def _parse_policy(policy: str) -> dict:
    """
    Parse 'CAUSE: X → Y' or 'PRECONDITION: X → Y' into cause/effect/event fields.
    Returns empty dict if pattern does not match.
    """
    m = re.match(r"(CAUSE|PRECONDITION):\s*(.+?)\s*→\s*(.+)", policy or "")
    if not m:
        return {}
    relation, cause, effect = m.group(1), m.group(2).strip(), m.group(3).strip()
    return {
        "cause": cause,
        "effect": effect,
        "event": effect,  # the effect trigger is the resulting domain event
    }


def auto_migrate_maven_ere(client: QdrantClient, collection: str):
    """Scroll all points and add cause/effect/event fields parsed from policy."""
    print(f"[auto] Migrating '{collection}'...")
    offset = None
    total = 0
    updated = 0

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for point in points:
            total += 1
            policy = point.payload.get("policy", "")
            new_fields = _parse_policy(policy)
            if new_fields:
                client.set_payload(
                    collection_name=collection,
                    payload=new_fields,
                    points=[point.id],
                )
                updated += 1

        print(f"  processed {total} points, updated {updated}...", end="\r")

        if next_offset is None:
            break
        offset = next_offset

    print(f"\n[auto] Done: {updated}/{total} points updated in '{collection}'")


# ---------------------------------------------------------------------------
# Manual mode: any collection, driven by JSON file
# ---------------------------------------------------------------------------

def manual_migrate(client: QdrantClient, updates_path: str):
    """
    Read manual_updates.json and apply set_payload to matching points.

    JSON format (list of update entries):
    [
      {
        "collection": "spring2026SE_g1_rag_mtop_commands",
        "match_text": "Remind me to start cooking dinner in 10 minutes",
        "payload": {
          "event": "Reminder Created",
          "command": "Create Reminder",
          "business_logic": "schedule reminder"
        }
      },
      ...
    ]

    match_text is matched against the 'document' field of each point.
    """
    with open(updates_path, "r", encoding="utf-8") as f:
        updates = json.load(f)

    print(f"[manual] Loaded {len(updates)} update entries from '{updates_path}'")

    applied = 0
    not_found = 0

    for entry in updates:
        collection = entry.get("collection", "")
        match_text = entry.get("match_text", "").strip()
        new_payload = entry.get("payload", {})

        if not collection or not match_text or not new_payload:
            print(f"  [skip] Invalid entry: {entry}")
            continue

        # Scroll to find matching point
        offset = None
        found_id = None
        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                doc = point.payload.get("document", "")
                if doc.strip() == match_text:
                    found_id = point.id
                    break
            if found_id or next_offset is None:
                break
            offset = next_offset

        if found_id:
            client.set_payload(
                collection_name=collection,
                payload=new_payload,
                points=[found_id],
            )
            print(f"  [ok] Updated point in '{collection}': {match_text[:60]}...")
            applied += 1
        else:
            print(f"  [not found] '{match_text[:60]}...' in '{collection}'")
            not_found += 1

    print(f"\n[manual] Done: {applied} applied, {not_found} not found")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant payloads: add new fields without re-embedding."
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-parse maven_ere_causal (cause/effect from policy field)",
    )
    parser.add_argument(
        "--manual",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to manual_updates.json for manual payload updates",
    )
    args = parser.parse_args()

    if not args.auto and not args.manual:
        parser.print_help()
        raise SystemExit(1)

    client = _get_client()

    if args.auto:
        maven_collection = f"{COLLECTION_PREFIX}maven_ere_causal"
        auto_migrate_maven_ere(client, maven_collection)

    if args.manual:
        manual_migrate(client, args.manual)

    print("\n=== Migration complete ===")
