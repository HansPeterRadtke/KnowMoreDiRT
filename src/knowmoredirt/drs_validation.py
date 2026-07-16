"""Pure structural validation helpers for model-produced DRS payloads."""

from __future__ import annotations

from typing import Any


def _item_id(item: dict[str, Any], key: str) -> str:
    return str(item.get(key) or "").strip()


def box_parent_cycle_errors(boxes: list[dict[str, Any]]) -> list[str]:
    """Return cycle errors for DRS box parent links.

    The caller reports missing parents and self-parent links; this helper
    catches multi-box subordinate loops.
    """

    box_ids = {_item_id(item, "id") for item in boxes if _item_id(item, "id")}
    graph: dict[str, list[str]] = {}
    for box in boxes:
        box_id = _item_id(box, "id")
        parent_id = _item_id(box, "parent_id")
        if not box_id:
            continue
        graph[box_id] = [parent_id] if parent_id in box_ids and parent_id != box_id else []

    errors: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(node: str) -> None:
        if node in visiting:
            start = stack.index(node) if node in stack else 0
            errors.append("cyclic_box_parent:" + "->".join([*stack[start:], node]))
            return
        if node in visited:
            return
        visiting.add(node)
        stack.append(node)
        for next_node in graph.get(node, []):
            visit(next_node)
            if errors:
                break
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for box_id in sorted(graph):
        visit(box_id)
        if len(errors) >= 50:
            break
    return errors[:50]


def box_root_errors(boxes: list[dict[str, Any]], *, require_asserted: bool = True) -> list[str]:
    """Return errors for DRS payloads without exactly one main box."""

    roots = [item for item in boxes if _item_id(item, "id") and not _item_id(item, "parent_id")]
    if not roots:
        return ["missing_root_box"]
    if len(roots) > 1:
        root_ids = ",".join(sorted(_item_id(item, "id") for item in roots))
        return [f"multiple_root_boxes:{root_ids}"]
    if require_asserted and _item_id(roots[0], "kind") != "asserted":
        return [f"bad_root_box_kind:{_item_id(roots[0], 'id')}:{_item_id(roots[0], 'kind')}"]
    return []


def condition_argument_cycle_errors(conditions: list[dict[str, Any]]) -> list[str]:
    """Return cycle errors for condition-to-condition DRS argument links."""

    condition_ids = {_item_id(item, "id") for item in conditions if _item_id(item, "id")}
    graph: dict[str, list[str]] = {}
    for condition in conditions:
        condition_id = _item_id(condition, "id")
        if not condition_id:
            continue
        edges: list[str] = []
        arguments = condition.get("arguments")
        arguments = [item for item in arguments if isinstance(item, dict)] if isinstance(arguments, list) else []
        for argument in arguments:
            target_id = _item_id(argument, "target_id")
            if _item_id(argument, "target_kind") == "condition" and target_id in condition_ids and target_id != condition_id:
                edges.append(target_id)
        graph[condition_id] = edges

    errors: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(node: str) -> None:
        if node in visiting:
            start = stack.index(node) if node in stack else 0
            errors.append("cyclic_condition_argument:" + "->".join([*stack[start:], node]))
            return
        if node in visited:
            return
        visiting.add(node)
        stack.append(node)
        for next_node in graph.get(node, []):
            visit(next_node)
            if errors:
                break
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for condition_id in sorted(graph):
        visit(condition_id)
        if len(errors) >= 50:
            break
    return errors[:50]
