"""Check the published experiment evidence without importing training code.

This checks only the files intentionally included in Git. It does not validate
omitted datasets, model weights, or per-example predictions and does not start
training. Requires Python 3.9+ and the standard library only.
"""

import ast
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / "provenance/source_files.json").read_text(encoding="utf-8"))
    paths = set()
    for item in manifest["files"]:
        relative = item["path"]
        if relative in paths:
            raise ValueError(f"Duplicate archived path: {relative}")
        paths.add(relative)
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f"Path outside archive: {relative}")
        content = path.read_bytes()
        if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"Archived file changed: {relative}")
        if path.suffix == ".py":
            ast.parse(content, filename=relative)

    hpo = ROOT / "hyperparameter_tuning"
    for stage, expected in (("search", 216), ("final", 72)):
        files = sorted((hpo / "records" / stage).glob("*.json"))
        if len(files) != expected:
            raise ValueError(f"Unexpected {stage} record count: {len(files)}")
        for file in files:
            record = json.loads(file.read_text(encoding="utf-8"))
            if record.get("status") != "completed":
                raise ValueError(f"Incomplete archived record: {file.name}")
        print(f"Completed {stage} records: {len(files)}")

    for folder in (ROOT / "outputs/summaries", hpo / "results/summaries"):
        for name, expected in (("all_test_metrics.csv", 936), ("three_seed_summary.csv", 312)):
            with (folder / name).open(encoding="utf-8", newline="") as stream:
                rows = sum(1 for _ in csv.DictReader(stream))
            if rows != expected:
                raise ValueError(f"Unexpected row count in {folder.name}/{name}: {rows}")
            print(f"{(folder / name).relative_to(ROOT)}: {rows} rows")

    print(f"PASS: {len(paths)} archived files match their recorded SHA-256 hashes.")
    print("Full predictions, weights, datasets and execution readiness are outside this check.")


if __name__ == "__main__":
    verify()
