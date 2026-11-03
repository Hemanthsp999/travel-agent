"""Save a compiled LangGraph workflow as Mermaid source, PNG and ASCII art.

`save_graph_artifacts` is called by every graph builder, so the workflow image is refreshed each
time an agent is built. Rendering problems are logged and never raised.
"""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

BOX_W, BOX_H, GAP_X, GAP_Y, MARGIN = 170, 38, 30, 80, 40


def save_graph_artifacts(
    compiled_graph: Any, name: str, out_dir: Path, remote_png: bool = True
) -> dict[str, Path]:
    """Write `<name>.mmd`, `<name>.png` and `<name>.txt` into `out_dir`; return what was written."""
    written: dict[str, Path] = {}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        graph = compiled_graph.get_graph(xray=True)
    except Exception as exc:
        logger.warning("Could not read graph for visualisation: %s", exc)
        return written

    _try(written, "mmd", out_dir / f"{name}.mmd", lambda p: p.write_text(graph.draw_mermaid()))
    _try(written, "png", out_dir / f"{name}.png", lambda p: p.write_bytes(_png_bytes(graph, remote_png)))
    _try(written, "txt", out_dir / f"{name}.txt", lambda p: p.write_text(graph.draw_ascii()))
    return written


def _try(written: dict[str, Path], kind: str, path: Path, action: Any) -> None:
    try:
        action(path)
        written[kind] = path
    except Exception as exc:
        path.unlink(missing_ok=True)
        logger.warning("Could not write %s artifact: %s", kind, exc)


def _png_bytes(graph: Any, remote: bool) -> bytes:
    if remote:
        try:
            return graph.draw_mermaid_png()
        except Exception as exc:
            logger.warning("Remote PNG rendering failed (%s); rendering locally", exc)
    return render_png_locally(graph)


# ------------------------------------------------------------------ local (offline) rendering


def _layers(node_ids: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    """Longest-path layering that ignores back edges, so loops do not break the layout."""
    outgoing: dict[str, list[str]] = {n: [] for n in node_ids}
    for src, dst in edges:
        outgoing.setdefault(src, []).append(dst)
    layer: dict[str, int] = {}
    on_stack: set[str] = set()

    def visit(node: str, depth: int) -> None:
        if depth <= layer.get(node, -1) or node in on_stack:
            return
        layer[node] = depth
        on_stack.add(node)
        for nxt in outgoing.get(node, []):
            visit(nxt, depth + 1)
        on_stack.discard(node)

    roots = [n for n in node_ids if n.startswith("__start__")] or node_ids[:1]
    for root in roots:
        visit(root, 0)
    for node in node_ids:
        layer.setdefault(node, 0)
    return layer


def render_png_locally(graph: Any) -> bytes:
    """Draw the graph with Pillow: boxes by layer, solid arrows, dashed arrows for conditional edges."""
    node_ids = list(graph.nodes)
    edges = [(e.source, e.target) for e in graph.edges]
    layer = _layers(node_ids, edges)
    rows: dict[int, list[str]] = {}
    for node in node_ids:
        rows.setdefault(layer[node], []).append(node)

    widest = max(len(r) for r in rows.values())
    width = MARGIN * 2 + widest * BOX_W + (widest - 1) * GAP_X
    height = MARGIN * 2 + len(rows) * BOX_H + (len(rows) - 1) * GAP_Y
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font()

    pos: dict[str, tuple[int, int]] = {}
    for depth, members in sorted(rows.items()):
        row_w = len(members) * BOX_W + (len(members) - 1) * GAP_X
        x0 = (width - row_w) // 2
        for i, node in enumerate(members):
            pos[node] = (x0 + i * (BOX_W + GAP_X), MARGIN + depth * (BOX_H + GAP_Y))

    for edge in graph.edges:
        if edge.source not in pos or edge.target not in pos:
            continue
        sx, sy = pos[edge.source]
        tx, ty = pos[edge.target]
        back = layer[edge.target] <= layer[edge.source]
        start = (sx + BOX_W // 2, sy + (0 if back else BOX_H))
        end = (tx + BOX_W // 2, ty + (BOX_H if back else 0))
        colour = "#c0392b" if back else "#555555"
        _line(draw, start, end, colour, dashed=bool(edge.conditional))
        _arrow_head(draw, start, end, colour)

    for node, (x, y) in pos.items():
        label = str(graph.nodes[node].name)
        special = node.startswith("__")
        draw.rounded_rectangle(
            (x, y, x + BOX_W, y + BOX_H), radius=8, fill="#e8f0fe" if not special else "#f1f1f1", outline="#1a73e8"
        )
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (BOX_W - (box[2] - box[0])) // 2, y + (BOX_H - (box[3] - box[1])) // 2 - 2), label, fill="black", font=font)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _font() -> Any:
    try:
        return ImageFont.load_default(size=14)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def _line(draw: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], colour: str, dashed: bool) -> None:
    if not dashed:
        draw.line([a, b], fill=colour, width=2)
        return
    steps = max(int(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 // 8), 1)
    for i in range(0, steps, 2):
        p = (a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps)
        q = (a[0] + (b[0] - a[0]) * (i + 1) / steps, a[1] + (b[1] - a[1]) * (i + 1) / steps)
        draw.line([p, q], fill=colour, width=2)


def _arrow_head(draw: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], colour: str) -> None:
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    size = 9
    left = (b[0] - size * math.cos(angle - 0.4), b[1] - size * math.sin(angle - 0.4))
    right = (b[0] - size * math.cos(angle + 0.4), b[1] - size * math.sin(angle + 0.4))
    draw.polygon([b, left, right], fill=colour)
