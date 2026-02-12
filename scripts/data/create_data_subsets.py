#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import random
import os


def read_json(path: Path) -> List[Dict[str, Any]]:
    """Reads either a JSON array (.json) or JSON Lines (.jsonl)."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # JSON array starts with '['.
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a top-level array.")
        return data


def write_json_array(path: Path, filename: str, records: List[Dict[str, Any]]) -> None:
    full_path = Path(path) / filename
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅  Done! {full_path} created with {len(records)} records.")


def infer_label(
    rec: Dict[str, Any],
    category_to_label: Dict[str, int],
    prefer_category_id: bool = True,
) -> int:
    """
    Determines integer label for a record.
    Priority:
    1) category_id (if prefer_category_id and present)
    2) category string mapping
    """
    if prefer_category_id and "category_id" in rec and rec["category_id"] is not None:
        try:
            return int(rec["category_id"])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid category_id {rec['category_id']!r}") from e

    cat = rec.get("category")
    if not isinstance(cat, str) or not cat.strip():
        raise ValueError("Record missing valid 'category' and no usable 'category_id'.")
    cat = cat.strip().lower()

    if cat not in category_to_label:
        raise ValueError(
            f"Unknown category {cat!r}. Known: {sorted(category_to_label.keys())}"
        )
    return category_to_label[cat]


def convert_records(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"Record {idx} is not an object.")

        if "text" not in rec or not isinstance(rec["text"], str) or not rec["text"].strip():
            raise ValueError(f"Record {idx} missing non-empty 'text' field.")

        label = int(rec["category_id"])

        new_rec: Dict[str, Any] = {"text": rec["text"], "label": label}
        out.append(new_rec)
    return out


def createEvalAndTestSets(records:List[Dict[str, Any]], output_path: Path, eval_filename: str, test_filename: str) -> None:
    grouped = defaultdict(list)
    for testdata in records:
        try:
            label = testdata["label"]
        except KeyError:
            print(f"⚠️ Warning: In Testdata missing 'label' field, skipping: {testdata}")
            continue
        if label is not None:  # Only add samples with valid labels
            grouped[label].append(testdata)

    print(f"📈 Found {len(grouped)} categories.")

    # Split each category 50/50
    eval_data = []
    test_data = []

    for label, group_data in sorted(grouped.items(), key=lambda x: (x[0] is None, x[0])):
        # Shuffle samples for this label before splitting
        random.shuffle(group_data)
        
        length_group = len(group_data)
        mid_point = length_group // 2
        
        eval_data.extend(group_data[:mid_point])
        test_data.extend(group_data[mid_point:])
        
        print(f"📈  Label {label}: {length_group} samples → {mid_point} eval, {length_group - mid_point} test")

    # Shuffle the final lists
    print("\n🔄 Shuffling final datasets again")
    random.shuffle(eval_data)
    random.shuffle(test_data)
    write_json_array(output_path,eval_filename, eval_data)
    write_json_array(output_path,test_filename, test_data)





def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_train_data", "-itr", required=True, help="Input .json file with training data")
    ap.add_argument("--input_test_data", "-ite", required=True, help="Input .json file with test data")
    ap.add_argument("--output_dataset_folder", "-o", required=True, help="Output folder for dataset files")

 
    args = ap.parse_args()
    itr = Path(args.input_train_data)
    ite = Path(args.input_test_data)
    outp = Path(args.output_dataset_folder)

    train_records = read_json(itr)
    test_records = read_json(ite)


    converted_train_records = convert_records(
        records=train_records,
    )
    converted_test_records = convert_records(
        records=test_records,
    )
    print(f"🔄 Found {len(converted_train_records)} train records.")
    print(f"🔄 Found {len(converted_test_records)} test records.")
    write_json_array(outp,"train_dataset.json", converted_train_records)
    createEvalAndTestSets(converted_test_records, outp, "eval_dataset.json", "test_dataset.json")


if __name__ == "__main__":
    main()
