#!/usr/bin/env python
"""Execute notebook code cells in order for local, non-GPU validation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    args = parser.parse_args()
    path = Path(args.notebook).resolve()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook_validation__"}
    original_cwd = Path.cwd()
    os.chdir(path.parents[1])
    try:
        for index, cell in enumerate(notebook["cells"], 1):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            print(f"\n=== Executing code cell {index} ===")
            exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
    finally:
        os.chdir(original_cwd)
    print(f"\nNotebook cell validation PASS: {path}")


if __name__ == "__main__":
    main()
