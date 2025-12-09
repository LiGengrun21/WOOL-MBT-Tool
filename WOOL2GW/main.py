#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

Batch-convert a directory of subfolders containing WOOL (+ optional JSON)
into a single GraphWalker JSON that contains multiple sub-models.

Folder layout expected:

<input_dir>/
  <sub1>/
    something.wool
    optional.json
  <sub2>/
    entrypoint_xxx.wool
    entrypoint_xxx.json   (recommended)
  <sub3>/
    root_self.wool
    context.json          (required)

The output file defaults to:
  <input_dir>/<input_dir.name>_gw.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import argparse
import json
import sys

from root import convert_root_context_to_gw
from wool2gw import convert_wool_to_gw


def _pick_json_file(folder: Path, wool_file: Path) -> Optional[Path]:
    """
    Heuristic to locate the most likely context JSON in a subfolder.

    Priority:
    1) <wool_stem>.json
    2) context.json
    3) the only *.json if there is exactly one
    4) otherwise None
    """
    stem_json = folder / f"{wool_file.stem}.json"
    if stem_json.exists():
        return stem_json

    ctx_json = folder / "context.json"
    if ctx_json.exists():
        return ctx_json

    all_json = sorted(folder.glob("*.json"))
    if len(all_json) == 1:
        return all_json[0]

    return None


def _pick_wool_file(folder: Path) -> Optional[Path]:
    wool_files = sorted(folder.glob("*.wool"))
    if not wool_files:
        return None
    # If multiple, pick the first in lexicographic order.
    return wool_files[0]


def convert_folder(
    input_dir: Path,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convert an input directory into a multi-model GraphWalker root JSON."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    models: List[Dict[str, Any]] = []
    warnings: List[str] = []

    subfolders = sorted([p for p in input_dir.iterdir() if p.is_dir()])

    for sub in subfolders:
        wool_file = _pick_wool_file(sub)
        if not wool_file:
            warnings.append(f"[SKIP] No .wool found in {sub}")
            continue

        json_file = _pick_json_file(sub, wool_file)

        lower_name = wool_file.name.lower()
        lower_stem = wool_file.stem.lower()

        try:
            if lower_name == "root_self.wool" or lower_stem.startswith("root_self"):
                if not json_file:
                    raise FileNotFoundError(
                        f"root_self requires a context json in {sub}"
                    )
                sub_root = convert_root_context_to_gw(
                    ctx_path=json_file,
                    wool_path=wool_file,
                    model_name=wool_file.stem,
                )

            elif lower_stem.startswith("entrypoint"):
                if not json_file:
                    warnings.append(
                        f"[WARN] No json for entrypoint wool {wool_file} in {sub}; converting without entry context"
                    )
                sub_root = convert_wool_to_gw(
                    wool_path=wool_file,
                    model_name=wool_file.stem,
                    entrypoint_context_path=json_file,
                )

            else:
                sub_root = convert_wool_to_gw(
                    wool_path=wool_file,
                    model_name=wool_file.stem,
                    entrypoint_context_path=None,
                )

            sub_models = sub_root.get("models", [])
            if not isinstance(sub_models, list) or not sub_models:
                warnings.append(f"[WARN] No models produced for {wool_file}")
                continue

            models.extend(sub_models)

        except Exception as e:
            warnings.append(f"[ERROR] {sub}: {e}")
            continue

    root_obj: Dict[str, Any] = {
        "models": models,
        "selectedModelIndex": 0,
        "selectedElementId": None,
    }

    # Write output if requested by caller.
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(root_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print warnings to stderr
    for w in warnings:
        print(w, file=sys.stderr)

    return root_obj


def main():
    ap = argparse.ArgumentParser(description="Batch convert subfolder WOOL files into a combined GraphWalker model")
    ap.add_argument("input_dir", type=Path, help="Input directory that contains subfolders")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output GW json path (default: <input_dir>/<input_dir.name>_gw.json)",
    )
    args = ap.parse_args()

    input_dir: Path = args.input_dir
    out_path: Path = args.out or (input_dir / f"{input_dir.name}_gw.json")

    convert_folder(input_dir, out_path=out_path)
    print(f"Saved combined GraphWalker model to {out_path}")


if __name__ == "__main__":
    main()
