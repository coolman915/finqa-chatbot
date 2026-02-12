#!/usr/bin/env python3
"""Upload FinQA as a LangSmith dataset."""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
from dotenv import load_dotenv
load_dotenv(_project_root / ".env", override=True)

from finqa_chatbot.config import get_settings
from finqa_chatbot.evaluation.langsmith_eval import upload_dataset


def main():
    parser = argparse.ArgumentParser(description="Upload FinQA dataset to LangSmith")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--name", type=str, default=None,
                        help="Dataset name (default: finqa-{split})")
    args = parser.parse_args()

    settings = get_settings()
    data_path = str(settings.dataset_dir / f"{args.split}.json")
    name = args.name or f"finqa-{args.split}"

    print(f"Uploading {data_path} as '{name}'...")
    dataset_id = upload_dataset(data_path, dataset_name=name)
    print(f"Dataset uploaded! ID: {dataset_id}")


if __name__ == "__main__":
    main()
