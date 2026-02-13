#!/usr/bin/env python3
"""Load FinQA JSON dataset files into MongoDB.

Usage:
    python scripts/load_dataset_mongo.py                  # load all splits
    python scripts/load_dataset_mongo.py --split dev      # load dev only
    python scripts/load_dataset_mongo.py --drop            # drop existing data first
"""

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from dotenv import load_dotenv
load_dotenv(_project_root / ".env", override=True)

from finqa_chatbot.config import get_settings
from finqa_chatbot.storage import get_mongo_store

ALL_SPLITS = ["train", "dev", "test", "private_test"]


def main():
    parser = argparse.ArgumentParser(description="Load FinQA dataset into MongoDB")
    parser.add_argument(
        "--split", choices=ALL_SPLITS, default=None,
        help="Load a single split (default: all)",
    )
    parser.add_argument(
        "--drop", action="store_true",
        help="Drop existing dataset entries before loading",
    )
    args = parser.parse_args()

    store = get_mongo_store()
    if store is None:
        print("ERROR: MongoDB not available. Set MONGODB_URI in .env")
        sys.exit(1)

    settings = get_settings()
    splits = [args.split] if args.split else ALL_SPLITS

    if args.drop:
        for split in splits:
            n = store.drop_dataset(split)
            print(f"Dropped {n} existing docs from split '{split}'")

    for split in splits:
        path = settings.dataset_dir / f"{split}.json"
        if not path.exists():
            print(f"  Skipping {split}: {path} not found")
            continue

        with open(path) as f:
            data = json.load(f)

        count = store.insert_dataset_entries(data, split)
        total = store.dataset.count_documents({"split": split})
        print(f"  {split}: loaded {len(data)} entries ({count} upserted, {total} total in DB)")

    print("\nDone.")
    store.close()


if __name__ == "__main__":
    main()
