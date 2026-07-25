#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.V01_leakage_probe_utils import run_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v01_split_view_dataset.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_probe(args.config, "b", args.seed), indent=2))


if __name__ == "__main__":
    main()

