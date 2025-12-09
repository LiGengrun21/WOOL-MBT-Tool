#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
root.py

Convert root-style WOOL context JSON to a GraphWalker model.

"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json, uuid, argparse, re

def uid() -> str:
    return str(uuid.uuid4())

@dataclass
class Vertex:
    id: str
    name: str
    properties: Dict[str, Any]
    sharedState: Optional[str] = None
    actions: Optional[List[str]] = None

    def to_json(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "properties": self.properties,
        }
        if self.sharedState is not None:
            data["sharedState"] = self.sharedState
        if self.actions:
            data["actions"] = self.actions
        return data

@dataclass
class Edge:
    id: str
    sourceVertexId: str
    targetVertexId: str
    name: Optional[str] = None
    actions: Optional[List[str]] = None
    guard: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "sourceVertexId": self.sourceVertexId,
            "targetVertexId": self.targetVertexId,
        }
        if self.name is not None:
            data["name"] = self.name
        if self.actions:
            data["actions"] = self.actions
        if self.guard is not None:
            data["guard"] = self.guard
        return data

@dataclass
class Model:
    id: str
    name: str
    startElementId: str
    generator: str = "random(edge_coverage(100))"
    edges: List[Edge] = field(default_factory=list)
    vertices: List[Vertex] = field(default_factory=list)
    actions: Optional[List[str]] = None

    def to_json(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "startElementId": self.startElementId,
            "generator": self.generator,
            "edges": [e.to_json() for e in self.edges],
            "vertices": [v.to_json() for v in self.vertices],
        }
        if self.actions:
            data["actions"] = self.actions
        return data

@dataclass
class GraphWalkerRoot:
    models: List[Model]
    selectedModelIndex: int = 0
    selectedElementId: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "models": [m.to_json() for m in self.models],
            "selectedModelIndex": self.selectedModelIndex,
            "selectedElementId": self.selectedElementId,
        }

def V(name: str,
      x: float,
      y: float,
      shared: Optional[str] = None,
      actions: Optional[List[str]] = None) -> Vertex:
    return Vertex(
        id=uid(),
        name=name,
        properties={"x": x, "y": y},
        sharedState=shared,
        actions=actions,
    )

def E(src: Vertex,
      tgt: Vertex,
      name: Optional[str] = None,
      actions: Optional[List[str]] = None,
      guard: Optional[str] = None) -> Edge:
    return Edge(
        id=uid(),
        sourceVertexId=src.id,
        targetVertexId=tgt.id,
        name=name,
        actions=actions,
        guard=guard,
    )

def _norm_label(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    return s if s else "Val"

YES_LIKE = {"Ja", "Yes"}
MAYBE_LIKE = {"Misschien", "Soms", "Maybe", "Sometimes"}
NO_LIKE = {"Nee", "No"}

def _make_skip_name(shared_dialog: str) -> str:
    base = shared_dialog
    if base.startswith("flag_condition_"):
        base = base[len("flag_condition_"):]
    if base.endswith("_instance"):
        base = base[:-len("_instance")]
    return f"v_skip_{base}"

# ---------------------------------------------------------------------
# Public, import-friendly API
# ---------------------------------------------------------------------

def convert_root_context_to_gw(
    ctx_path: Path,
    wool_path: Optional[Path] = None,
    model_name: str = "root_model",
) -> Dict[str, Any]:
    """
    Convert a root-style context JSON to a GraphWalker root JSON dict.

    Parameters
    ----------
    ctx_path:
        Path to context JSON (must contain key "template").
    wool_path:
        Kept for API parity with the other converter. This converter
        doesn't currently need the wool content.
    model_name:
        Name of the produced model.

    Returns
    -------
    A GraphWalker root JSON dict with a single model.
    """
    return build_converter(ctx_path, wool_path or Path("."), model_name=model_name)

# Backward-compatible alias
def build_converter(
    ctx_path: Path,
    wool_path: Path,
    model_name: str = "root_model",
) -> Dict[str, Any]:
    t = json.loads(ctx_path.read_text(encoding="utf-8"))["template"]
    flags_all = t.get("flags", [])
    # 过滤 NEVERTRIGGERED
    effective_flags = [
        f for f in flags_all
        if "NEVERTRIGGERED" not in set(f.get("trigvals", []))
    ]
    entry_shared = t["entrypoint_default"]["n"].split(".")[0]

    vertices: List[Vertex] = []
    edges: List[Edge] = []

    # --- 基础顶点 ---
    v_start = V(
        "v_start",
        -31.72584219868196,
        271.97622713698087,
        "v_start_heredity",
        [
            "global.DISCUSSED_CONTACT_PROFESSIONAL = false;",
            "global.return_from_opt_out = false;",
            "global.return_from_explain = false;",
        ],
    )
    v_entry = V(
        f"v_{entry_shared}",
        -16.01453434384885,
        165.90644830111847,
        f"v_{entry_shared}",
    )
    v_flags_end = V("v_flags_end", 1058.0586622809506, 278, None)

    vertices.extend([v_start, v_entry, v_flags_end])

    decision_vertices: List[Vertex] = []
    spec_vertices: List[Vertex] = []
    spec_end_vertices: List[Vertex] = []
    skip_vertices: List[Vertex] = []

    for idx, f in enumerate(effective_flags, start=1):
        vx = {1: 166, 2: 438, 3: 710}.get(idx, 166 + (idx - 1) * 272)
        vy = 270 if idx != 3 else 276
        init_action = "global.flag1=0;" if idx == 1 else f"global.flag{idx}= 0;"
        v_flag = V(f"v_flag_{idx}", vx, vy, None, [init_action])

        shared_dialog = f["n"].split(".")[0]

        # random position numbers for display in GW studio
        spec_x = {
            1: 1271.832050205671,
            2: 804.473917507188,
            3: 443.73541900039845,
        }.get(idx, 700.0)
        spec_y = {
            1: 177.46434070547133,
            2: 175.93979796585444,
            3: 169.6307507466054,
        }.get(idx, 170.0)
        spec = V(
            f"v_{shared_dialog}",
            spec_x,
            spec_y,
            f"v_{shared_dialog}",
        )

        end_x = {
            1: 1320.9669886178085,
            2: 798.3742067955247,
            3: 321.0808953907722,
        }.get(idx, 710.0)
        end_y = {
            1: -137.0586622809509,
            2: -128.02377286301981,
            3: -127.51188643150999,
        }.get(idx, 20.0)
        spec_end = V(
            f"v_{shared_dialog}_end",
            end_x,
            end_y,
            f"v_{shared_dialog}_end",
        )

        skip_name = _make_skip_name(shared_dialog)
        skip_x = {
            1: 1381.0,
            2: 808.0,
            3: 410.6085176367823,
        }.get(idx, 400.0 + 200 * idx)
        skip_y = {
            1: 50.0,
            2: 26.999999999999915,
            3: 17.999999999999957,
        }.get(idx, 0.0)
        v_skip = V(skip_name, skip_x, skip_y, skip_name)

        vertices.extend([v_flag, spec, spec_end, v_skip])
        decision_vertices.append(v_flag)
        spec_vertices.append(spec)
        spec_end_vertices.append(spec_end)
        skip_vertices.append(v_skip)

    if decision_vertices:
        edges.append(E(v_start, decision_vertices[0]))

    for i, v_flag in enumerate(decision_vertices, start=1):
        next_target = decision_vertices[i] if i < len(decision_vertices) else v_flags_end
        trig_raw = list(effective_flags[i - 1].get("trigvals", []))

        built_any = False
        built_no = False
        for tv in trig_raw:
            if tv == "NEVERTRIGGERED":
                continue
            label = _norm_label(tv)
            if tv in NO_LIKE:
                edges.append(
                    E(
                        v_flag,
                        next_target,
                        f"e_{label}_{i}",
                        [f"global.flag{i}= 0;"],
                    )
                )
                built_no = True
                built_any = True
            elif tv in YES_LIKE or tv in MAYBE_LIKE:
                edges.append(
                    E(
                        v_flag,
                        next_target,
                        f"e_{label}_{i}",
                        [f"global.flag{i}= 1;"],
                    )
                )
                built_any = True
            else:
                edges.append(
                    E(
                        v_flag,
                        next_target,
                        f"e_{label}_{i}",
                        [f"global.flag{i}= 1;"],
                    )
                )
                built_any = True

        if i == 1 and "Soms" not in trig_raw:
            edges.append(
                E(
                    v_flag,
                    next_target,
                    "e_Soms_1",
                    ["global.flag1= 1;"],
                )
            )
            built_any = True

        if not built_no:
            edges.append(
                E(
                    v_flag,
                    next_target,
                    f"e_Nee_{i}",
                    [f"global.flag{i}= 0;"],
                )
            )
        if not built_any:
            edges.append(
                E(
                    v_flag,
                    next_target,
                    f"e_Ja_{i}",
                    [f"global.flag{i}= 1;"],
                )
            )

    n_flags = len(spec_vertices)
    if n_flags >= 1:
        edges.append(
            E(
                v_flags_end,
                spec_vertices[0],
                guard="global.flag1==1",
            )
        )
        edges.append(
            E(
                v_flags_end,
                spec_end_vertices[0],
                guard="global.flag1== 0",
            )
        )

        for i in range(1, n_flags):
            spec_end_i = spec_end_vertices[i - 1]
            spec_next = spec_vertices[i]
            spec_end_next = spec_end_vertices[i]

            is_last_flag = (i + 1 == n_flags)
            if is_last_flag:
                guard_yes = f"global.flag{i+1}==1"
                guard_no = f"global.flag{i+1}==0"
            else:
                guard_yes = f"global.flag{i+1}== 1"
                guard_no = f"global.flag{i+1}== 0"

            edges.append(
                E(
                    spec_end_i,
                    spec_next,
                    guard=guard_yes,
                )
            )
            edges.append(
                E(
                    spec_end_i,
                    spec_end_next,
                    guard=guard_no,
                )
            )

        last_end = spec_end_vertices[-1]
        edges.append(
            E(
                last_end,
                v_entry,
            )
        )

    for i, (v_skip, spec, spec_end) in enumerate(
        zip(skip_vertices, spec_vertices, spec_end_vertices),
        start=1,
    ):
        # “Ja, ik wil ze overslaan.” -> end
        edges.append(
            E(
                v_skip,
                spec_end,
                "Ja, ik wil ze overslaan.",
            )
        )

        edges.append(
            E(
                v_skip,
                spec,
                "Nee, stel de vragen dan toch maar.",
                actions=["global.return_from_opt_out = true;"],
            )
        )

    model_actions = [
        "global.DISCUSSED_CONTACT_PROFESSIONAL = false;",
        "global.return_from_opt_out = false;",
        "global.return_from_explain = false;",
    ] + [f"global.flag{i} = 0;" for i in range(1, len(effective_flags) + 1)]

    model = Model(
        id=uid(),
        name=model_name,
        startElementId=v_start.id,
        edges=edges,
        vertices=vertices,
        actions=model_actions,
    )
    root = GraphWalkerRoot(models=[model], selectedModelIndex=0, selectedElementId=None)
    return root.to_json()

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True, type=Path, help="Path to context JSON")
    ap.add_argument("--wool", required=True, type=Path, help="Path to WOOL file (not used, kept for parity)")
    ap.add_argument("--out", required=True, type=Path, help="Output GraphWalker JSON path")
    ap.add_argument("--model-name", default="root_model", help="Model name")
    args = ap.parse_args()

    root = convert_root_context_to_gw(args.context, args.wool, model_name=args.model_name)
    args.out.write_text(json.dumps(root, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved GraphWalker model to {args.out}")

if __name__ == "__main__":
    main()
