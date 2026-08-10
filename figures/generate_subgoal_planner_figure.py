from __future__ import annotations

import math
from pathlib import Path


WIDTH = 1600
HEIGHT = 920

BG = "#FFFFFF"
PANEL_BG = "#FFFFFF"
PANEL_BORDER = "#C9CFD6"
TEXT = "#22262B"
SUBTEXT = "#5C6670"
FLOW = "#7A848F"
LIGHT_LINE = "#D8DEE5"

TRAVERSABLE = "#C7DAB6"
NON_TRAVERSABLE = "#E8EBEF"
GOAL_DIR = "#F2E2A0"

EMBED_FILL = "#D6E3DD"
ENCODER_FILL = "#EDF3F1"
FILTER_FILL = "#E9EFF6"
EFF = "#7FCB9A"
SAFE = "#D98C83"
FUSION_FILL = "#F3F0E8"
SUBGOAL = "#8EBDEB"
CANDIDATE = "#C9D5CA"

FONT = "Inter, Arial, Helvetica, sans-serif"


def svg_header() -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" fill="none">\n'
        "  <defs>\n"
        '    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">\n'
        f'      <path d="M 0 0 L 10 5 L 0 10 z" fill="{FLOW}"/>\n'
        "    </marker>\n"
        '    <style>\n'
        f'      .title {{ font: 700 26px {FONT}; fill: {TEXT}; }}\n'
        f'      .panel-title {{ font: 700 24px {FONT}; fill: {TEXT}; }}\n'
        f'      .label {{ font: 600 17px {FONT}; fill: {TEXT}; }}\n'
        f'      .small {{ font: 500 15px {FONT}; fill: {SUBTEXT}; }}\n'
        f'      .chip {{ font: 600 14px {FONT}; fill: {TEXT}; }}\n'
        f'      .stage {{ font: 700 16px {FONT}; fill: {TEXT}; letter-spacing: 0.02em; }}\n'
        "    </style>\n"
        "  </defs>\n"
    )


def rr(x: float, y: float, w: float, h: float, r: float, fill: str, stroke: str, sw: float = 1.5) -> str:
    return (
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>\n'
    )


def txt(x: float, y: float, text: str, cls: str, anchor: str = "start") -> str:
    return f'  <text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{text}</text>\n'


def line(x1: float, y1: float, x2: float, y2: float, dashed: bool = False, arrow: bool = False, sw: float = 2.0) -> str:
    dash = ' stroke-dasharray="7 7"' if dashed else ""
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    return (
        f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{FLOW}" '
        f'stroke-width="{sw}" stroke-linecap="round"{dash}{marker}/>\n'
    )


def path(d: str, fill: str = "none", stroke: str = FLOW, sw: float = 2.0, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="7 7"' if dashed else ""
    return f'  <path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"{dash}/>\n'


def circle(cx: float, cy: float, r: float, fill: str, stroke: str = "none", sw: float = 1.0) -> str:
    return f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>\n'


def wedge(cx: float, cy: float, r: float, start_deg: float, end_deg: float, fill: str) -> str:
    start = math.radians(start_deg)
    end = math.radians(end_deg)
    x1 = cx + r * math.cos(start)
    y1 = cy + r * math.sin(start)
    x2 = cx + r * math.cos(end)
    y2 = cy + r * math.sin(end)
    large = 1 if abs(end_deg - start_deg) > 180 else 0
    d = f"M {cx} {cy} L {x1:.2f} {y1:.2f} A {r} {r} 0 {large} 1 {x2:.2f} {y2:.2f} Z"
    return path(d, fill=fill, stroke=LIGHT_LINE, sw=2.0)


def robot_icon(cx: float, cy: float, scale: float = 1.0) -> str:
    s = scale
    parts = [
        rr(cx - 20 * s, cy - 26 * s, 40 * s, 42 * s, 8 * s, "#FFFFFF", TEXT, 2.0),
        rr(cx - 10 * s, cy - 42 * s, 20 * s, 10 * s, 5 * s, "#FFFFFF", TEXT, 2.0),
        line(cx - 6 * s, cy - 32 * s, cx - 6 * s, cy - 26 * s, sw=2.0),
        line(cx + 6 * s, cy - 32 * s, cx + 6 * s, cy - 26 * s, sw=2.0),
        circle(cx - 7 * s, cy - 7 * s, 3.3 * s, TEXT),
        circle(cx + 7 * s, cy - 7 * s, 3.3 * s, TEXT),
        rr(cx - 8 * s, cy + 4 * s, 16 * s, 7 * s, 3 * s, "#FFFFFF", TEXT, 2.0),
        line(cx - 20 * s, cy - 2 * s, cx - 32 * s, cy + 5 * s, sw=2.0),
        line(cx + 20 * s, cy - 2 * s, cx + 32 * s, cy + 5 * s, sw=2.0),
        line(cx - 8 * s, cy + 16 * s, cx - 18 * s, cy + 27 * s, sw=2.0),
        line(cx + 8 * s, cy + 16 * s, cx + 18 * s, cy + 27 * s, sw=2.0),
    ]
    return "".join(parts)


def flag_icon(x: float, y: float, scale: float = 1.0) -> str:
    s = scale
    return "".join(
        [
            line(x, y - 20 * s, x, y + 24 * s, sw=2.2),
            path(
                f"M {x} {y - 18 * s} "
                f"L {x + 22 * s} {y - 12 * s} "
                f"L {x + 12 * s} {y + 2 * s} "
                f"L {x} {y - 2 * s} Z",
                fill="#FFFFFF",
                stroke=TEXT,
                sw=2.0,
            ),
            circle(x, y - 22 * s, 3.3 * s, TEXT),
        ]
    )


def neural_head(cx: float, cy: float, color: str) -> str:
    nodes = [
        (cx - 26, cy - 22),
        (cx - 26, cy),
        (cx - 26, cy + 22),
        (cx, cy - 28),
        (cx, cy),
        (cx, cy + 28),
        (cx + 28, cy - 16),
        (cx + 28, cy + 16),
    ]
    edges = [
        (0, 3), (0, 4), (1, 3), (1, 4), (1, 5), (2, 4), (2, 5),
        (3, 6), (4, 6), (4, 7), (5, 7)
    ]
    chunks = []
    for a, b in edges:
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]
        chunks.append(
            f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            'stroke-width="2.1" stroke-linecap="round"/>\n'
        )
    for x, y in nodes:
        chunks.append(circle(x, y, 4.5, "#FFFFFF", color, 2.2))
    return "".join(chunks)


def chevron(x: float, y: float, w: float, h: float) -> str:
    d = (
        f"M {x} {y} "
        f"L {x + w * 0.55} {y} "
        f"L {x + w} {y + h / 2} "
        f"L {x + w * 0.55} {y + h} "
        f"L {x} {y + h} "
        f"L {x + w * 0.42} {y + h / 2} Z"
    )
    return path(d, fill="#FFFFFF", stroke=PANEL_BORDER, sw=1.8)


def add_left_panel(parts: list[str]) -> None:
    x, y, w, h = 40, 70, 730, 780
    parts.append(rr(x, y, w, h, 28, PANEL_BG, PANEL_BORDER, 1.7))
    parts.append(txt(x + 28, y + 44, "Candidate Subgoal Generation", "panel-title"))
    parts.append(line(x + 28, y + 62, x + w - 28, y + 62, sw=1.4))

    legend_y = y + 92
    legend = [
        ("Traversable", TRAVERSABLE),
        ("Blocked", NON_TRAVERSABLE),
        ("Goal Dir.", GOAL_DIR),
    ]
    lx = x + 28
    for label, fill in legend:
        parts.append(rr(lx, legend_y - 18, 18, 18, 4, fill, LIGHT_LINE, 1.2))
        parts.append(txt(lx + 28, legend_y - 3, label, "small"))
        lx += 165

    cx, cy, r = x + 285, y + 378, 218
    sector_defs = [
        (-162, -132, TRAVERSABLE),
        (-132, -104, NON_TRAVERSABLE),
        (-104, -76, TRAVERSABLE),
        (-76, -49, NON_TRAVERSABLE),
        (-49, -24, GOAL_DIR),
        (-24, 2, NON_TRAVERSABLE),
        (2, 30, TRAVERSABLE),
    ]
    for start, end, fill in sector_defs:
        parts.append(wedge(cx, cy, r, start, end, fill))

    parts.append(path(
        f"M {cx - r} {cy} A {r} {r} 0 0 1 {cx + r * math.cos(math.radians(30)):.2f} {cy + r * math.sin(math.radians(30)):.2f}",
        stroke=LIGHT_LINE,
        sw=2.0,
    ))
    parts.append(robot_icon(cx, cy + 3, 0.86))

    goal_x = x + 610
    goal_y = y + 152
    parts.append(flag_icon(goal_x, goal_y, 1.0))
    parts.append(line(cx, cy + 4, goal_x - 14, goal_y + 12, dashed=True, arrow=False, sw=2.0))

    init_label_y = y + 548
    parts.append(txt(x + 28, init_label_y, "Initial Candidate Set", "stage"))
    parts.append(rr(x + 240, init_label_y - 29, 385, 48, 16, "#FFFFFF", PANEL_BORDER, 1.4))
    candidate_xs = [x + 270 + i * 42 for i in range(8)]
    candidate_fills = [CANDIDATE] * 7 + [GOAL_DIR]
    candidate_rs = [11, 11, 11, 10, 10, 11, 11, 11]
    for cx_i, fill, rad in zip(candidate_xs, candidate_fills, candidate_rs):
        parts.append(circle(cx_i, init_label_y - 6, rad, fill, "#FFFFFF", 1.2))
    parts.append(txt(x + 538, init_label_y - 5, "...", "label", "middle"))
    parts.append(line(x + 430, init_label_y + 28, x + 430, init_label_y + 82, arrow=True, sw=2.0))

    filter_y = y + 634
    parts.append(txt(x + 353, filter_y - 28, "Filter", "stage", "middle"))
    filter_blocks = [
        (x + 42, "Goal Alignment"),
        (x + 246, "Local Clearance"),
        (x + 468, "Angular Diversity"),
    ]
    for bx, label in filter_blocks:
        parts.append(rr(bx, filter_y - 8, 180, 46, 14, FILTER_FILL, "#D2DAE5", 1.2))
        parts.append(txt(bx + 90, filter_y + 20, label, "chip", "middle"))

    parts.append(line(x + 430, filter_y + 46, x + 430, filter_y + 104, arrow=True, sw=2.0))
    top_y = y + 716
    parts.append(txt(x + 28, top_y, "Top-M Candidates", "stage"))
    parts.append(rr(x + 240, top_y - 29, 318, 48, 16, "#FFFFFF", PANEL_BORDER, 1.4))
    top_xs = [x + 274 + i * 42 for i in range(6)]
    top_fills = [CANDIDATE, CANDIDATE, CANDIDATE, CANDIDATE, CANDIDATE, GOAL_DIR]
    for cx_i, fill in zip(top_xs, top_fills):
        parts.append(circle(cx_i, top_y - 6, 11, fill, "#FFFFFF", 1.2))
    parts.append(txt(x + 454, top_y - 5, "...", "label", "middle"))


def add_right_panel(parts: list[str]) -> None:
    x, y, w, h = 830, 70, 730, 780
    parts.append(rr(x, y, w, h, 28, PANEL_BG, PANEL_BORDER, 1.7))
    parts.append(txt(x + 28, y + 44, "Value-Based Subgoal Scoring", "panel-title"))
    parts.append(line(x + 28, y + 62, x + w - 28, y + 62, sw=1.4))

    embed_specs = [
        (x + 148, y + 98, "LiDAR Embedding"),
        (x + 148, y + 165, "Goal Embedding"),
        (x + 148, y + 232, "Candidate Embedding"),
    ]
    for bx, by, label in embed_specs:
        parts.append(rr(bx, by, 252, 48, 14, EMBED_FILL, "#C0CDC6", 1.2))
        parts.append(txt(bx + 126, by + 31, label, "label", "middle"))

    parts.append(line(x + 274, y + 282, x + 274, y + 336, arrow=True, sw=2.0))

    enc_x, enc_y, enc_w, enc_h = x + 176, y + 344, 196, 92
    parts.append(rr(enc_x, enc_y, enc_w, enc_h, 20, ENCODER_FILL, "#C4D1CA", 1.5))
    parts.append(txt(enc_x + enc_w / 2, enc_y + 54, "Shared Encoder", "label", "middle"))
    parts.append(path(
        f"M {enc_x + 20} {enc_y + 24} "
        f"L {enc_x + 46} {enc_y + 38} "
        f"L {enc_x + 20} {enc_y + 52} "
        f"L {enc_x + 52} {enc_y + 66} "
        f"M {enc_x + enc_w - 20} {enc_y + 24} "
        f"L {enc_x + enc_w - 46} {enc_y + 38} "
        f"L {enc_x + enc_w - 20} {enc_y + 52} "
        f"L {enc_x + enc_w - 52} {enc_y + 66}",
        stroke="#92A49B",
        sw=2.0,
    ))

    branch_y = y + 505
    parts.append(line(enc_x + 46, enc_y + enc_h, x + 195, branch_y - 50, arrow=True, sw=2.0))
    parts.append(line(enc_x + enc_w - 46, enc_y + enc_h, x + 535, branch_y - 50, arrow=True, sw=2.0))

    parts.append(neural_head(x + 210, branch_y - 2, EFF))
    parts.append(txt(x + 210, branch_y + 74, "Efficiency Head", "label", "middle"))
    parts.append(neural_head(x + 520, branch_y - 2, SAFE))
    parts.append(txt(x + 520, branch_y + 74, "Safety Head", "label", "middle"))

    parts.append(line(x + 210, branch_y + 92, x + 330, y + 636, arrow=True, sw=2.0))
    parts.append(line(x + 520, branch_y + 92, x + 400, y + 636, arrow=True, sw=2.0))

    fusion_x, fusion_y, fusion_w, fusion_h = x + 272, y + 640, 146, 48
    parts.append(rr(fusion_x, fusion_y, fusion_w, fusion_h, 14, FUSION_FILL, "#D8D1C5", 1.3))
    parts.append(txt(fusion_x + fusion_w / 2, fusion_y + 31, "Fusion Score", "label", "middle"))

    parts.append(line(fusion_x + fusion_w / 2, fusion_y + fusion_h, fusion_x + fusion_w / 2, y + 735, arrow=True, sw=2.0))
    parts.append(rr(x + 297, y + 734, 96, 30, 12, "#F7FAFD", "#D3E0EE", 1.3))
    parts.append(circle(x + 319, y + 749, 8.5, SUBGOAL))
    parts.append(txt(x + 347, y + 755, "Subgoal", "label"))


def build_svg() -> str:
    parts: list[str] = [svg_header(), rr(0, 0, WIDTH, HEIGHT, 0, BG, BG, 0)]
    parts.append(txt(WIDTH / 2, 38, "Subgoal Planner", "title", "middle"))
    add_left_panel(parts)
    parts.append(chevron(781, 392, 28, 44))
    add_right_panel(parts)
    parts.append("</svg>\n")
    return "".join(parts)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "subgoal_planner_publication.svg"
    out_path.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
