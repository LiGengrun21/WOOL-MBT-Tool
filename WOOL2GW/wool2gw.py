#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
wool2gw.py

Convert a WOOL dialogue file to a GraphWalker model JSON.

"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import argparse
import json
import re
import uuid


# ---------- WOOL AST ----------

@dataclass
class WoolOption:
    text: str          # 按钮文案（如果 [[Target]] 则为空字符串）
    target: str        # 跳转目标 title
    guard: Optional[str] = None  # 由 <<if ...>> / <<else>> 转成的条件表达式
    actions: List[str] = field(default_factory=list)  # 行内 <<set>> 对应的 edge.actions


@dataclass
class WoolNode:
    title: str
    tags: List[str]
    speaker: Optional[str]
    color_id: Optional[int]
    position: Optional[tuple[float, float]]
    body_lines: List[str]
    options: List[WoolOption] = field(default_factory=list)
    node_actions: List[str] = field(default_factory=list)  # 不跟 option 绑定的 <<set>>
    output: Optional[Any] = None  # 纯展示文本；可能是 str 或 {cond: text}


@dataclass
class WoolDialogue:
    nodes: Dict[str, WoolNode]


def _extract_output_from_body_lines(body_lines: List[str]) -> Optional[str]:
    """从 body 去掉 <<...>> 和 [[...]]，留下纯文本。"""
    cleaned_lines: List[str] = []
    for line in body_lines:
        # 去掉命令块和选项块
        no_cmd = re.sub(r"<<.*?>>", "", line)
        no_opt = re.sub(r"\[\[.*?\]\]", "", no_cmd)
        text = no_opt.strip()
        if text:
            cleaned_lines.append(text)
    if not cleaned_lines:
        return None
    return "\n".join(cleaned_lines)


def parse_wool(text: str) -> WoolDialogue:
    """
    parse .wool to WoolDialogue

    """
    sections = [s.strip() for s in text.split("===") if s.strip()]
    nodes: Dict[str, WoolNode] = {}

    for sec in sections:
        lines = [l.rstrip("\n") for l in sec.splitlines()]

        try:
            sep_idx = lines.index("---")
        except ValueError:
            continue

        header_lines = lines[:sep_idx]
        body_lines = lines[sep_idx + 1:]

        header: Dict[str, str] = {}
        for hl in header_lines:
            if ":" in hl:
                k, v = hl.split(":", 1)
                header[k.strip().lower()] = v.strip()

        title = header.get("title")
        if not title:
            continue

        tags_str = header.get("tags", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str is not None else []
        speaker = header.get("speaker") or None

        color_id: Optional[int] = None
        if "colorid" in header and header["colorid"]:
            try:
                color_id = int(header["colorid"])
            except ValueError:
                pass

        position: Optional[tuple[float, float]] = None
        if "position" in header and header["position"]:
            try:
                px, py = header["position"].split(",", 1)
                position = (float(px), float(py))
            except Exception:
                position = None

        options: List[WoolOption] = []
        node_actions: List[str] = []

        # guard：[(cond_raw, cond_gw, mode)]  mode: "if" | "else"
        guard_stack: List[tuple[str, str, str]] = []

        def _convert_condition_pair(cond: str) -> tuple[str, str]:
            raw = cond.strip().rstrip(";")
            gw = raw.replace("$", "global.").replace("!==", "!=").replace("===", "==")
            return raw, gw

        def _negate(expr: str) -> str:
            s = expr.strip()
            if s.startswith("(") and s.endswith(")"):
                s_inner = s[1:-1].strip()
            else:
                s_inner = s
            m = re.match(r"^(.*?)(==|!=)(.*)$", s_inner)
            if m:
                left, op, right = m.group(1).strip(), m.group(2), m.group(3).strip()
                return f"{left} != {right}" if op == "==" else f"{left} == {right}"
            return f"!({expr})"

        def current_guard_gw() -> Optional[str]:
            if not guard_stack:
                return None
            parts = []
            for cond_raw, cond_gw, mode in guard_stack:
                parts.append(cond_gw if mode == "if" else _negate(cond_gw))
            return " && ".join(parts)

        def current_guard_raw() -> Optional[str]:
            if not guard_stack:
                return None
            parts = []
            for cond_raw, cond_gw, mode in guard_stack:
                parts.append(cond_raw if mode == "if" else _negate(cond_raw))
            return " && ".join(parts)

        unconditional_lines: List[str] = []
        conditional_map: Dict[str, List[str]] = {}

        in_each_questions = False

        for line in body_lines:
            stripped = line.strip()

            if stripped.startswith("{{#each questions"):
                in_each_questions = True
                continue

            if stripped.startswith("{{/each"):
                in_each_questions = False
                continue

            if in_each_questions:
                continue

            m_if = re.match(r"<<\s*if\s+(.*?)\s*>>", stripped)
            m_elseif = re.match(r"<<\s*elseif\s+(.*?)\s*>>", stripped)
            m_else = re.match(r"<<\s*else\s*>>", stripped)
            m_endif = re.match(r"<<\s*endif\s*>>", stripped)

            if m_if:
                raw, gw = _convert_condition_pair(m_if.group(1))
                guard_stack.append((raw, gw, "if"))
                continue

            if m_elseif:
                raw, gw = _convert_condition_pair(m_elseif.group(1))
                if guard_stack:
                    guard_stack.pop()
                guard_stack.append((raw, gw, "if"))
                continue

            if m_else:
                if guard_stack:
                    raw, gw, _ = guard_stack.pop()
                    guard_stack.append((raw, gw, "else"))
                continue

            if m_endif:
                if guard_stack:
                    guard_stack.pop()
                continue

            line_guard_gw = current_guard_gw()
            line_guard_raw = current_guard_raw()

            # --- extract output text ---
            no_cmd = re.sub(r"<<.*?>>", "", line)
            no_opt = re.sub(r"\[\[.*?\]\]", "", no_cmd)
            text_out = no_opt.strip()
            if text_out:
                if line_guard_raw is None:
                    unconditional_lines.append(text_out)
                else:
                    conditional_map.setdefault(line_guard_raw, []).append(text_out)

            # --- parse options  ---
            line_options: List[WoolOption] = []
            for inner in re.findall(r"\[\[(.*?)\]\]", line):
                inner = inner.strip()
                if not inner:
                    continue

                parts = [p.strip() for p in inner.split("|")]
                if len(parts) == 1:
                    label, target, extra_parts = parts[0], parts[0], []
                else:
                    label, target, extra_parts = parts[0], parts[1], parts[2:]

                edge_label = label if label and label != target else ""
                opt = WoolOption(text=edge_label, target=target, guard=line_guard_gw)

                for extra in extra_parts:
                    for sm in re.findall(r"<<\s*set\s+(.*?)\s*>>", extra):
                        m_set = re.match(r"(?:\$(\w+)|global\.(\w+))\s*=\s*(.*)", sm)
                        if not m_set:
                            continue
                        var_name = m_set.group(1) or m_set.group(2)
                        expr = m_set.group(3).strip().replace("$", "global.")
                        opt.actions.append(f"global.{var_name} = {expr};")

                options.append(opt)
                line_options.append(opt)

            line_for_sets = re.sub(r"\[\[(.*?)\]\]", "", line)
            for sm in re.findall(r"<<\s*set\s+(.*?)\s*>>", line_for_sets):
                m_set = re.match(r"(?:\$(\w+)|global\.(\w+))\s*=\s*(.*)", sm)
                if not m_set:
                    continue
                var_name = m_set.group(1) or m_set.group(2)
                expr = m_set.group(3).strip().replace("$", "global.")
                action = f"global.{var_name} = {expr};"

                if line_options:
                    line_options[-1].actions.append(action)
                else:
                    node_actions.append(action)

        # --- build output ---
        if conditional_map:
            base_uncond = " ".join(unconditional_lines).strip() if unconditional_lines else ""
            output: Optional[Any] = {}
            for cond_raw, texts in conditional_map.items():
                t = " ".join(texts).strip()
                if base_uncond:
                    t = (base_uncond + " " + t).strip()
                output[cond_raw] = t
        else:
            output = " ".join(unconditional_lines).strip() if unconditional_lines else None

        nodes[title] = WoolNode(
            title=title,
            tags=tags,
            speaker=speaker,
            color_id=color_id,
            position=position,
            body_lines=body_lines,
            options=options,
            node_actions=node_actions,
            output=output,
        )

    return WoolDialogue(nodes=nodes)


# ---------- GraphWalker structure ----------
@dataclass
class GWVertex:
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
class GWEdge:
    id: str
    sourceVertexId: str
    targetVertexId: str
    name: Optional[str] = None
    guard: Optional[str] = None
    actions: Optional[List[str]] = None

    def to_json(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "sourceVertexId": self.sourceVertexId,
            "targetVertexId": self.targetVertexId,
        }
        if self.name:
            data["name"] = self.name
        if self.guard:
            data["guard"] = self.guard
        if self.actions:
            data["actions"] = self.actions
        return data


@dataclass
class GWModel:
    id: str
    name: str
    startElementId: str
    generator: str = "random(edge_coverage(100))"
    vertices: List[GWVertex] = field(default_factory=list)
    edges: List[GWEdge] = field(default_factory=list)
    actions: Optional[List[str]] = None

    def to_json(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "startElementId": self.startElementId,
            "generator": self.generator,
            "vertices": [v.to_json() for v in self.vertices],
            "edges": [e.to_json() for e in self.edges],
        }
        if self.actions:
            data["actions"] = self.actions
        return data


@dataclass
class GWRoot:
    models: List[GWModel]
    selectedModelIndex: int = 0
    selectedElementId: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "models": [m.to_json() for m in self.models],
            "selectedModelIndex": self.selectedModelIndex,
            "selectedElementId": self.selectedElementId,
        }


# ---------- other functions ----------

def _derive_end_shared_name_from_stem(stem: str) -> str:
    return f"v_{stem}_end"


def _derive_skip_suffix_from_flag_stem(stem: str) -> str:
    base = stem
    if base.startswith("flag_condition_"):
        base = base[len("flag_condition_"):]
    if base.endswith("_instance"):
        base = base[:-len("_instance")]
    if base.endswith("health") and len(base) > len("health"):
        return base[:-len("health")] + "_health"

    return base


def wool_to_graphwalker(
    dialogue: WoolDialogue,
    model_name: str,
    end_shared_name: str,
    ignore_special_nodes: bool,
    is_flag_model: bool,
    skip_shared_name: Optional[str],
    start_shared_name: Optional[str],
    entry_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    special_titles = {"start", "overview", "branchnode"}

    # 1) choose the nodes to keep
    if ignore_special_nodes:
        kept_nodes: Dict[str, WoolNode] = {
            title: node
            for title, node in dialogue.nodes.items()
            if title.lower() not in special_titles
        }
    else:
        kept_nodes = dict(dialogue.nodes)

    # 2) check if any .Skip target exists
    has_skip_target = False
    if is_flag_model:
        for node in kept_nodes.values():
            for opt in node.options:
                if opt.target.endswith(".Skip"):
                    has_skip_target = True
                    break
            if has_skip_target:
                break

    skip_vertex_id: Optional[str] = None
    if is_flag_model and has_skip_target and skip_shared_name is not None:
        skip_vertex_id = str(uuid.uuid4())

    # 3) map node title to vertex id
    node_to_vertex_id: Dict[str, str] = {
        title: str(uuid.uuid4()) for title in kept_nodes.keys()
    }

    # 4) generate edges
    edges: List[GWEdge] = []
    indegree: Dict[str, int] = {title: 0 for title in kept_nodes.keys()}
    assigned_vars: set[str] = set()

    def _collect_vars_from_actions(actions: List[str]) -> None:
        for act in actions:
            m = re.match(r"\s*global\.([A-Za-z0-9_]+)\s*=", act)
            if m:
                assigned_vars.add(m.group(1))

    for title, node in kept_nodes.items():
        src_id = node_to_vertex_id[title]
        for opt in node.options:
            target_title = opt.target

            if is_flag_model and target_title.endswith(".Skip") and skip_vertex_id is not None:
                tgt_id = skip_vertex_id
            elif target_title in node_to_vertex_id:
                tgt_id = node_to_vertex_id[target_title]
                indegree[target_title] += 1
            else:
                continue

            actions = opt.actions if opt.actions else None
            if actions:
                _collect_vars_from_actions(actions)

            edges.append(
                GWEdge(
                    id=str(uuid.uuid4()),
                    sourceVertexId=src_id,
                    targetVertexId=tgt_id,
                    name=(opt.text or None),
                    guard=opt.guard,
                    actions=actions,
                )
            )

    # 5) generate vertices
    vertices: List[GWVertex] = []
    idx_kept = 0
    start_vertex_id: Optional[str] = None

    for title, node in kept_nodes.items():
        if node.position:
            x, y = node.position
        else:
            x = (idx_kept % 5) * 300
            y = (idx_kept // 5) * 200

        shared: Optional[str] = None
        if title == "End":
            shared = end_shared_name
        elif title == "Start" and start_shared_name is not None:
            shared = start_shared_name
        elif title.startswith("Response") and indegree.get(title, 0) == 0:
            shared = f"v_{title}"

        v_actions = node.node_actions if node.node_actions else None
        if v_actions:
            _collect_vars_from_actions(v_actions)

        props = {"x": x, "y": y}
        if node.output is not None:
            props["output"] = node.output

        v = GWVertex(
            id=node_to_vertex_id[title],
            name=f"v_{title}",
            properties=props,
            sharedState=shared,
            actions=v_actions,
        )
        vertices.append(v)
        idx_kept += 1

        if title == "Start":
            start_vertex_id = v.id

    # 6)  v_Skip
    if skip_vertex_id is not None:
        vertices.append(
            GWVertex(
                id=skip_vertex_id,
                name="v_Skip",
                properties={"x": 100.0, "y": 100.0},
                sharedState=skip_shared_name,
            )
        )

    # ----------  entrypoint context dynamic extension ----------
    model_actions: List[str] = []

    if entry_ctx:
        max_top = int(entry_ctx.get("maxTopQuestions", 0))
        questions = entry_ctx.get("questions", []) or []
        q_count = len(questions)

        #  model actions init
        model_actions.append(f"global.maxTopQuestions = {max_top};")
        for i in range(1, q_count + 1):
            model_actions.append(f"global.rq_{i}_asked = false;")
            model_actions.append(f"global.rq_{i}_top = 0;")

        def _find_title_case_insensitive(tname: str) -> Optional[str]:
            if tname in node_to_vertex_id:
                return tname
            for t in node_to_vertex_id.keys():
                if t.lower() == tname.lower():
                    return t
            return None

        def _ensure_vertex_for_title(tname: str, x: float, y: float) -> str:
            real_title = _find_title_case_insensitive(tname)
            if real_title:
                return node_to_vertex_id[real_title]

            node_to_vertex_id[tname] = str(uuid.uuid4())
            vertices.append(
                GWVertex(
                    id=node_to_vertex_id[tname],
                    name=f"v_{tname}",
                    properties={"x": x, "y": y},
                )
            )
            indegree[tname] = indegree.get(tname, 0)
            return node_to_vertex_id[tname]

        showtop_id = _ensure_vertex_for_title("ShowTop", 0.0, 0.0)
        archive_id = _ensure_vertex_for_title("QuestionArchive", 600.0, 0.0)
        other_id = _ensure_vertex_for_title("OtherQuestions", 300.0, 0.0)

        base_x = 0.0
        base_y = (idx_kept // 5 + 1) * 200

        for idx, qitem in enumerate(questions, start=1):
            q_text = str(qitem.get("q", "")).strip()
            n_title = str(qitem.get("n", "")).strip()
            if not n_title:
                continue

            if n_title not in node_to_vertex_id:
                node_to_vertex_id[n_title] = str(uuid.uuid4())
                vx = base_x + (idx - 1) * 260
                vy = base_y

                vname = f"v_{n_title}"
                vertices.append(
                    GWVertex(
                        id=node_to_vertex_id[n_title],
                        name=vname,
                        properties={"x": vx, "y": vy},
                        sharedState=vname, 
                    )
                )
                indegree[n_title] = indegree.get(n_title, 0)

            dyn_id = node_to_vertex_id[n_title]

            guard_showtop = (
                f"global.rq_{idx}_top - global.aq < global.maxTopQuestions "
                f"&& !global.rq_{idx}_asked"
            )
            actions_showtop = [
                "global.topq = global.topq + 1;",
                f"global.rq_{idx}_asked = true;",
                "global.aq = global.aq + 1;",
            ]
            edges.append(
                GWEdge(
                    id=str(uuid.uuid4()),
                    sourceVertexId=showtop_id,
                    targetVertexId=dyn_id,
                    name=q_text or None,
                    guard=guard_showtop,
                    actions=actions_showtop,
                )
            )
            indegree[n_title] = indegree.get(n_title, 0) + 1

            guard_from_archive = f"global.rq_{idx}_asked == true"
            edges.append(
                GWEdge(
                    id=str(uuid.uuid4()),
                    sourceVertexId=archive_id,
                    targetVertexId=dyn_id,
                    name=q_text or None,
                    guard=guard_from_archive,
                )
            )
            indegree[n_title] = indegree.get(n_title, 0) + 1

            indegree_key = "QuestionArchive" if "QuestionArchive" in indegree else _find_title_case_insensitive("QuestionArchive")
            if indegree_key:
                indegree[indegree_key] = indegree.get(indegree_key, 0) + 1

            guard_other = (
                f"!global.rq_{idx}_asked && "
                f"global.rq_{idx}_top >= global.maxTopQuestions"
            )
            actions_other = [
                f"global.rq_{idx}_asked = true;"
            ]
            edges.append(
                GWEdge(
                    id=str(uuid.uuid4()),
                    sourceVertexId=other_id,
                    targetVertexId=dyn_id,
                    name=q_text or None,
                    guard=guard_other,
                    actions=actions_other,
                )
            )
            indegree[n_title] = indegree.get(n_title, 0) + 1


    model = GWModel(
        id=str(uuid.uuid4()),
        name=model_name,
        startElementId="",
        vertices=vertices,
        edges=edges,
        actions=model_actions or None,
    )

    root = GWRoot(models=[model])
    return root.to_json()

def load_entrypoint_context(entrypoint_context_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not entrypoint_context_path:
        return None
    if not entrypoint_context_path.exists():
        return None
    return json.loads(entrypoint_context_path.read_text(encoding="utf-8"))

def preprocess_wool_text(text: str, entry_ctx: Optional[Dict[str, Any]]) -> str:
    if not entry_ctx:
        return text

    questions = entry_ctx.get("questions", []) or []
    q_len = len(questions)
    max_top = entry_ctx.get("maxTopQuestions", 0)

    text = text.replace("{{questions.length}}", str(q_len))
    text = text.replace("{{maxTopQuestions}}", str(max_top))
    text = text.replace("{{../maxTopQuestions}}", str(max_top))
    return text

def convert_wool_to_gw(
    wool_path: Path,
    model_name: Optional[str] = None,
    entrypoint_context_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Convert a single WOOL file to a GraphWalker root JSON dict.

    Parameters
    ----------
    wool_path:
        Path to .wool.
    model_name:
        If None, defaults to wool filename stem.
    entrypoint_context_path:
        Optional context for entrypoint* wool.

    Returns
    -------
    A GraphWalker root JSON dict with a single model.
    """
    entry_ctx = load_entrypoint_context(entrypoint_context_path)

    text = wool_path.read_text(encoding="utf-8")
    text = preprocess_wool_text(text, entry_ctx)

    dialogue = parse_wool(text)

    stem = wool_path.stem
    final_model_name = model_name or stem

    end_shared_name = _derive_end_shared_name_from_stem(stem)

    ignore_special_nodes = stem.lower().startswith("info")

    is_flag_model = stem.lower().startswith("flag")
    is_entrypoint_model = stem.lower().startswith("entrypoint")

    skip_shared_name: Optional[str] = None
    start_shared_name: Optional[str] = None

    if is_flag_model:
        suffix = _derive_skip_suffix_from_flag_stem(stem)
        skip_shared_name = f"v_skip_{suffix}"

    if is_flag_model or is_entrypoint_model:
        start_shared_name = f"v_{stem}"

    gw_root = wool_to_graphwalker(
        dialogue=dialogue,
        model_name=final_model_name,
        end_shared_name=end_shared_name,
        ignore_special_nodes=ignore_special_nodes,
        is_flag_model=is_flag_model,
        skip_shared_name=skip_shared_name,
        start_shared_name=start_shared_name,
        entry_ctx=entry_ctx,
    )

    return gw_root


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert WOOL dialogue to GraphWalker model JSON."
    )
    parser.add_argument("--wool", required=True, type=Path, help="Path to .wool file")
    parser.add_argument("--out", required=True, type=Path, help="Output GraphWalker JSON path")
    parser.add_argument(
        "--model-name",
        required=False,
        default=None,
        help="GraphWalker model name (default: <wool filename without extension>)",
    )
    parser.add_argument(
        "--entrypoint-context",
        required=False,
        type=Path,
        default=None,
        help="Optional entrypoint context JSON to expand questions/top logic",
    )

    args = parser.parse_args()

    gw_root = convert_wool_to_gw(
        wool_path=args.wool,
        model_name=args.model_name,
        entrypoint_context_path=args.entrypoint_context,
    )

    args.out.write_text(json.dumps(gw_root, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved GraphWalker model to {args.out}")


if __name__ == "__main__":
    main()
