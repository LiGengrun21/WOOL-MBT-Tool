#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Python step functions from a GraphWalker JSON model,
with:
1) properties.output -> assert generation in vertex functions
2) actions -> global variable assignments translated to Python
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Helpers: naming
# ----------------------------

def sanitize_ascii(s: Optional[str]) -> str:
    if not s:
        s = "Unnamed"
    nfkd = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "Unnamed"
    if re.match(r"^\d", s):
        s = "_" + s
    return s


def short_hash(s: str, n: int = 4) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def normalize_vertex_name(name: Optional[str]) -> str:
    if not name:
        return "Vertex"
    return name[2:] if name.startswith("v_") else name


def normalize_edge_name(name: Optional[str]) -> str:
    if not name:
        return "Edge"
    return name[2:] if name.startswith("e_") else name


def unique_func_name(prefix: str, base: str, id_str: str, used: set) -> str:
    for n in (4, 6, 8, 10, 12):
        suf = short_hash(id_str, n)
        fname = f"{prefix}_{base}_{suf}"
        if fname not in used:
            used.add(fname)
            return fname
    fname = f"{prefix}_{base}_{sanitize_ascii(id_str)}"
    if fname in used:
        i = 2
        while f"{fname}_{i}" in used:
            i += 1
        fname = f"{fname}_{i}"
    used.add(fname)
    return fname


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Vertex:
    id: str
    name: str = ""
    sharedState: str = ""
    actions: List[str] = None
    properties: Dict[str, Any] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Vertex":
        return Vertex(
            id=d["id"],
            name=d.get("name", "") or "",
            sharedState=d.get("sharedState", "") or "",
            actions=d.get("actions", []) or [],
            properties=d.get("properties", {}) or {},
        )


@dataclass
class Edge:
    id: str
    sourceVertexId: str
    targetVertexId: str
    name: str = ""
    guard: str = ""
    actions: List[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Edge":
        return Edge(
            id=d["id"],
            sourceVertexId=d["sourceVertexId"],
            targetVertexId=d["targetVertexId"],
            name=d.get("name", "") or "",
            guard=d.get("guard", "") or "",
            actions=d.get("actions", []) or [],
        )


@dataclass
class Model:
    id: str
    name: str
    vertices: List[Vertex]
    edges: List[Edge]
    actions: List[str]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Model":
        return Model(
            id=d["id"],
            name=d.get("name", "") or "",
            vertices=[Vertex.from_dict(v) for v in (d.get("vertices", []) or [])],
            edges=[Edge.from_dict(e) for e in (d.get("edges", []) or [])],
            actions=d.get("actions", []) or [],
        )


# ----------------------------
# Action parsing / translation
# ----------------------------

_GLOBAL_ASSIGN_RE = re.compile(r"^\s*global\.([A-Za-z0-9_]+)\s*=\s*(.+?)\s*;\s*$")

def _normalize_value_token(tok: str) -> str:
    t = tok.strip()
    if t == "true":
        return "True"
    if t == "false":
        return "False"
    # number?
    if re.fullmatch(r"-?\d+(\.\d+)?", t):
        return t
    # quoted string?
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        return t
    return t  # expression or variable

def gw_expr_to_py(expr: str) -> str:
    """
    Convert a GW-ish expression into Python expression focusing on:
    - global.xxx -> ctx.globals.get("xxx")
    - $xxx -> ctx.globals.get("xxx")
    - &&, ||, !==, ===
    This is intentionally lightweight.
    """
    e = expr.strip()

    # operators
    e = e.replace("&&", " and ")
    e = e.replace("||", " or ")
    e = e.replace("!==", "!=")
    e = e.replace("===", "==")

    # Replace global.xxx
    def repl_global(m):
        var = m.group(1)
        return f'ctx.globals.get("{var}")'
    e = re.sub(r"\bglobal\.([A-Za-z0-9_]+)\b", repl_global, e)

    # Replace $var tokens
    def repl_dollar(m):
        var = m.group(1)
        return f'ctx.globals.get("{var}")'
    e = re.sub(r"\$([A-Za-z0-9_]+)", repl_dollar, e)

    # Normalize booleans if present
    e = re.sub(r"\btrue\b", "True", e)
    e = re.sub(r"\bfalse\b", "False", e)

    return e

def parse_actions_to_python(actions: List[str]) -> List[str]:
    """
    Translate a subset of GW actions into Python lines.
    """
    if not actions:
        return []

    lines: List[str] = []
    for raw in actions:
        if not raw or not raw.strip():
            continue
        m = _GLOBAL_ASSIGN_RE.match(raw)
        if not m:
            lines.append(f"# TODO: Unparsed action: {raw}")
            continue

        var = m.group(1)
        rhs = m.group(2).strip()

        # Basic expression translation:
        # convert global.xxx refs inside rhs
        rhs_py = gw_expr_to_py(rhs)

        # Allow simple literals to remain clean
        rhs_py = _normalize_value_token(rhs_py)

        lines.append(f'ctx.globals["{var}"] = {rhs_py}')
    return lines


# ----------------------------
# Output parsing -> asserts
# ----------------------------

def gen_output_assert_block(output_val: Any, indent: str = "    ") -> List[str]:
    lines: List[str] = []
    if output_val is None:
        return lines

    if isinstance(output_val, str):
        txt = output_val.strip()
        if not txt:
            return lines
        lines.append(f'{indent}_assert_output_equals(ctx, {txt!r})')
        return lines

    if isinstance(output_val, dict):
        items = list(output_val.items())
        if not items:
            return lines

        for idx, (cond, text) in enumerate(items):
            cond_py = gw_expr_to_py(cond)
            kw = "if" if idx == 0 else "elif"
            lines.append(f"{indent}{kw} {cond_py}:")
            if isinstance(text, str):
                lines.append(f"{indent}    _assert_output_equals(ctx, {text!r})")
            else:
                lines.append(f"{indent}    # TODO: non-string output")
                lines.append(f"{indent}    pass")

        lines.append(f"{indent}else:")
        lines.append(f"{indent}    pass")
        return lines

    lines.append(f"{indent}# TODO: Unsupported output type: {type(output_val).__name__}")
    return lines



# ----------------------------
# Code generation
# ----------------------------

def gen_vertex_func(v: Vertex, fname: str, model_name: str) -> str:
    props = v.properties or {}
    output_val = props.get("output")

    doc = [
        f"Vertex: {v.name}",
        f"id: {v.id}",
        f"model: {model_name}",
    ]
    if v.sharedState:
        doc.append(f"sharedState: {v.sharedState}")
    if v.actions:
        doc.append("actions:")
        doc.extend([f"  - {a}" for a in v.actions])
    if "output" in props:
        doc.append("output: (auto-assert generated when possible)")

    docstring = "\\n".join(doc)

    # actions -> python
    act_lines = parse_actions_to_python(v.actions)

    # output -> asserts
    assert_lines = gen_output_assert_block(output_val, indent="    ")

    body_lines: List[str] = []
    if act_lines:
        body_lines.append("    # --- actions (translated from GW) ---")
        body_lines.extend([f"    {ln}" for ln in act_lines])
        body_lines.append("")

    if assert_lines:
        body_lines.append("    # --- output asserts (from properties.output) ---")
        body_lines.extend(assert_lines)
        body_lines.append("")

    if not body_lines:
        body_lines.append("    pass")

    body = "\n".join(body_lines).rstrip()

    return f'''
def {fname}(ctx):
    """
    {docstring}
    """
{body}
'''.lstrip()


def gen_edge_func(e: Edge, fname: str, model_name: str) -> str:
    doc = [
        f"Edge: {e.name or '<unnamed>'}",
        f"id: {e.id}",
        f"model: {model_name}",
        f"sourceVertexId: {e.sourceVertexId}",
        f"targetVertexId: {e.targetVertexId}",
    ]
    if e.guard:
        doc.append(f"guard: {e.guard}")
    if e.actions:
        doc.append("actions:")
        doc.extend([f"  - {a}" for a in e.actions])

    docstring = "\\n".join(doc)

    act_lines = parse_actions_to_python(e.actions)

    body_lines: List[str] = []
    if act_lines:
        body_lines.append("    # --- actions (translated from GW) ---")
        body_lines.extend([f"    {ln}" for ln in act_lines])
        body_lines.append("")

    body_lines.append("    pass")

    body = "\n".join(body_lines).rstrip()

    return f'''
def {fname}(ctx):
    """
    {docstring}
    """
{body}
'''.lstrip()


def build_name_maps(models: List[Model]) -> Tuple[
    Dict[str, str], Dict[str, str],
    Dict[str, List[str]], Dict[str, List[str]],
]:
    used = set()
    vertex_id_to_func: Dict[str, str] = {}
    edge_id_to_func: Dict[str, str] = {}
    vertex_name_to_funcs: Dict[str, List[str]] = defaultdict(list)
    edge_name_to_funcs: Dict[str, List[str]] = defaultdict(list)

    for m in models:
        for v in m.vertices:
            raw = normalize_vertex_name(v.name)
            base = sanitize_ascii(raw)
            fname = unique_func_name("v", base, v.id, used)
            vertex_id_to_func[v.id] = fname
            vertex_name_to_funcs[v.name].append(fname)

        for e in m.edges:
            raw = normalize_edge_name(e.name)
            base = sanitize_ascii(raw)
            fname = unique_func_name("e", base, e.id, used)
            edge_id_to_func[e.id] = fname
            edge_name_to_funcs[e.name].append(fname)

    return vertex_id_to_func, edge_id_to_func, vertex_name_to_funcs, edge_name_to_funcs


def collect_declared_globals(models: List[Model]) -> List[str]:
    """
    Heuristically collect global vars that appear on LHS of actions
    so you can see what's defined in the model. 
    """
    vars_set = set()

    def scan_actions(actions: List[str]):
        for a in actions or []:
            m = _GLOBAL_ASSIGN_RE.match(a or "")
            if m:
                vars_set.add(m.group(1))

    for m in models:
        scan_actions(m.actions)
        for v in m.vertices:
            scan_actions(v.actions)
        for e in m.edges:
            scan_actions(e.actions)

    return sorted(vars_set)


def generate_steps_module(models: List[Model]) -> str:
    v_id_to_func, e_id_to_func, v_name_to_funcs, e_name_to_funcs = build_name_maps(models)
    declared_globals = collect_declared_globals(models)

    out: List[str] = []
    out.append("# -*- coding: utf-8 -*-")
    out.append('"""\nAuto-generated GraphWalker steps.\n'
               "Includes:\n"
               "  - actions -> ctx.globals assignments\n"
               "  - properties.output -> assert generation when possible\n"
               '"""\n')

    out.append("""
class StepContext:
    \"\"\"Minimal placeholder context for generated steps.

    Expected:
      ctx.globals: dict for GW 'global' variables
      ctx.runtime: dict for runtime info (e.g., last_output)
    \"\"\"
    def __init__(self, globals=None, model=None, runtime=None):
        self.globals = globals if globals is not None else {}
        self.model = model
        self.runtime = runtime if runtime is not None else {}
""".lstrip())

    out.append("""
def _get_last_output(ctx) -> str:
    \"\"\"Fetch last output from runtime/global fallback.\"\"\"
    if isinstance(getattr(ctx, "runtime", None), dict) and "last_output" in ctx.runtime:
        return ctx.runtime.get("last_output") or ""
    if isinstance(getattr(ctx, "globals", None), dict):
        return ctx.globals.get("_last_output") or ""
    return ""

def _assert_output_equals(ctx, expected: str):
    actual = _get_last_output(ctx)
    assert actual == expected, f"Output mismatch. expected={expected!r}, actual={actual!r}"
""".lstrip())

    # Declared global vars summary
    if declared_globals:
        out.append("\n# ----------------------------")
        out.append("# Globals detected from actions")
        out.append("# ----------------------------")
        for gv in declared_globals:
            out.append(f'# global.{gv}')
        out.append("")

    # Emit model-level init actions (as comments + optional helper)
    for m in models:
        if m.actions:
            out.append(f"\n# Model init actions for: {m.name}")
            for a in m.actions:
                out.append(f"#   {a}")

    # Vertices grouped by model
    for m in models:
        out.append(f"\n# ----------------------------")
        out.append(f"# Vertices for model: {m.name}")
        out.append(f"# ----------------------------\n")
        for v in m.vertices:
            fname = v_id_to_func[v.id]
            out.append(gen_vertex_func(v, fname, m.name))

    # Edges grouped by model
    for m in models:
        out.append(f"\n# ----------------------------")
        out.append(f"# Edges for model: {m.name}")
        out.append(f"# ----------------------------\n")
        for e in m.edges:
            fname = e_id_to_func[e.id]
            out.append(gen_edge_func(e, fname, m.name))

    # Lookup tables
    out.append("\n# ----------------------------")
    out.append("# Lookup tables")
    out.append("# ----------------------------\n")

    out.append("VERTEX_ID_TO_FUNC = {")
    for vid, fname in sorted(v_id_to_func.items()):
        out.append(f'    "{vid}": {fname},')
    out.append("}\n")

    out.append("EDGE_ID_TO_FUNC = {")
    for eid, fname in sorted(e_id_to_func.items()):
        out.append(f'    "{eid}": {fname},')
    out.append("}\n")

    out.append("VERTEX_NAME_TO_FUNCS = {")
    for name, fnames in sorted(v_name_to_funcs.items(), key=lambda x: x[0]):
        safe_list = ", ".join(fnames)
        out.append(f'    {name!r}: [{safe_list}],')
    out.append("}\n")

    out.append("EDGE_NAME_TO_FUNCS = {")
    for name, fnames in sorted(e_name_to_funcs.items(), key=lambda x: x[0]):
        safe_list = ", ".join(fnames)
        out.append(f'    {name!r}: [{safe_list}],')
    out.append("}\n")

    return "\n".join(out)


def load_models(json_path: str) -> List[Model]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    models_data = data.get("models", []) or []
    return [Model.from_dict(m) for m in models_data]


def main():
    parser = argparse.ArgumentParser(description="Generate ASCII-safe Python steps from GW model JSON.")
    parser.add_argument("input", help="Path to GW model JSON")
    parser.add_argument("-o", "--output", default="gw_steps.py", help="Output Python module path")
    args = parser.parse_args()

    models = load_models(args.input)
    code = generate_steps_module(models)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
