# Maintenance utility — delete Qdrant collections by prefix or name
#
# Usage:
#   python reset_collections.py --prefix bpc_        # delete all bpc_* collections
#   python reset_collections.py --prefix maven_ere   # delete maven_ere_causal
#   python reset_collections.py --all                # delete everything
#   python reset_collections.py --list               # just list, no deletion

import os
import argparse
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def list_collections(client: QdrantClient) -> list[str]:
    return [c.name for c in client.get_collections().collections]


def delete_by_prefix(client: QdrantClient, prefix: str) -> list[str]:
    targets = [n for n in list_collections(client) if n.startswith(prefix)]
    for name in targets:
        client.delete_collection(collection_name=name)
        print(f"  Deleted: {name}")
    return targets


def delete_all(client: QdrantClient) -> list[str]:
    targets = list_collections(client)
    for name in targets:
        client.delete_collection(collection_name=name)
        print(f"  Deleted: {name}")
    return targets


def main():
    parser = argparse.ArgumentParser(description="Reset Qdrant collections")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prefix", metavar="PREFIX",
                       help="Delete all collections whose name starts with PREFIX")
    group.add_argument("--all", action="store_true",
                       help="Delete ALL collections")
    group.add_argument("--list", action="store_true",
                       help="List collections without deleting")
    args = parser.parse_args()

    client = QdrantClient(url=QDRANT_URL, timeout=300)
    collections = list_collections(client)

    if args.list:
        if not collections:
            print("No collections found.")
        for name in collections:
            count = client.get_collection(name).points_count
            print(f"  {name}  ({count} vectors)")
        return

    if not collections:
        print("No collections found. Nothing to delete.")
        return

    print("Current collections:")
    for name in collections:
        print(f"  {name}")
    print()

    if args.all:
        confirm = input("Delete ALL collections? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return
        deleted = delete_all(client)
    else:
        targets = [n for n in collections if n.startswith(args.prefix)]
        if not targets:
            print(f"No collections match prefix '{args.prefix}'. Nothing to delete.")
            return
        confirm = input(f"Delete {len(targets)} collection(s) with prefix '{args.prefix}'? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return
        deleted = delete_by_prefix(client, args.prefix)

    print(f"\nDone. Deleted {len(deleted)} collection(s).")


if __name__ == "__main__":
    main()
