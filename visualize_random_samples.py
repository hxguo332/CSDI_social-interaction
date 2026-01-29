"""
Sample random trajectory segments per scene and save individual visualizations.

Each "segment" is defined by splitting on time gaps (> gap_minutes, default 5).
For each scene, we shuffle all segments and take up to N=100, render each to a
separate PNG under output_dir/{scene}/sample_{idx}.png.

Usage:
  python visualize_random_samples.py --output samples --per-scene 100 --gap-minutes 5
"""

from __future__ import annotations

import argparse
import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import csv
import math
from PIL import Image, ImageDraw, ImageFont

try:
    import cairosvg
except ImportError as exc:  # pragma: no cover - runtime import guard
    raise SystemExit("cairosvg is required. Install: pip install cairosvg") from exc

ROOT = Path(__file__).resolve().parent

SCENES: Dict[str, Dict[str, Any]] = {
    "german_1": {
        "svg": ROOT / "dataset_RTLS/german_1/german_1.svg",
        "data": ROOT / "dataset_RTLS/german_1/german_1.txt",
        "origin_px": (314.45606625, 1037.2104925),
        "scale_px_per_m": (56.6829, 56.6829),
    },
    "german_2": {
        "svg": ROOT / "dataset_RTLS/german_2/german_2_ref.svg",
        "data": ROOT / "dataset_RTLS/german_2/german_2.txt",
        "origin_px": (629.70825, 403.09325),
        "scale_px_per_m": (36.4965147, 35.8203997),
    },
    "german_3": {
        "svg": ROOT / "dataset_RTLS/german_3/german_3.svg",
        "data": ROOT / "dataset_RTLS/german_3/german_3.txt",
        "origin_px": (515.0, 754.0),
        "scale_px_per_m": (14.1952, 14.1952),
    },
    "german_4": {
        "svg": ROOT / "dataset_RTLS/german_4/german_4.svg",
        "data": ROOT / "dataset_RTLS/german_4/german_4.txt",
        "origin_px": (160.0, 272.0),
        "scale_px_per_m": (36.4172, 36.4172),
    },
}


def load_all_segments(data_path: Path, gap_minutes: float) -> List[Dict[str, Any]]:
    per_tag: Dict[str, List[Tuple[datetime, float, float]]] = {}
    with data_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            tag = row.get("tag_id")
            if not tag:
                continue
            try:
                x = float(row.get("x", "nan"))
                y = float(row.get("y", "nan"))
                t = datetime.fromisoformat(row.get("time"))
            except Exception:
                continue
            per_tag.setdefault(tag, []).append((t, x, y))

    gap = timedelta(minutes=gap_minutes)
    segments: List[Dict[str, Any]] = []
    for tag, rows in per_tag.items():
        rows.sort(key=lambda r: r[0])
        current: List[Tuple[datetime, float, float]] = []
        last_t: datetime | None = None
        start_t: datetime | None = None
        for t, x, y in rows:
            if last_t is None or t - last_t <= gap:
                current.append((t, x, y))
                start_t = start_t or t
            else:
                if current:
                    segments.append({"tag": tag, "start": start_t, "end": last_t, "pts": current})
                current = [(t, x, y)]
                start_t = t
            last_t = t
        if current:
            segments.append({"tag": tag, "start": start_t, "end": last_t, "pts": current})

    segments.sort(key=lambda s: s["start"])
    return segments


def convert_pts(
    pts: List[Tuple[datetime, float, float]],
    origin_px: Tuple[float, float],
    scale: Tuple[float, float],
) -> Tuple[List[Tuple[float, float]], List[float], List[float]]:
    """
    Convert (t,x,y) list to pixel coords and compute speed (m/s) per segment step.
    speeds length equals len(pts)-1, for coloring lines; first point speed set equal to first segment.
    """
    ox, oy = origin_px
    sx, sy = scale
    pts_px: List[Tuple[float, float]] = []
    speeds: List[float] = []
    dts: List[float] = []
    last_t = None
    last_xy = None
    for t, x, y in pts:
        px = ox + x * sx
        py = oy + y * sy  # no flip-y per request
        pts_px.append((px, py))
        if last_t is not None and last_xy is not None:
            dt = (t - last_t).total_seconds()
            if dt > 0:
                dist = math.hypot(x - last_xy[0], y - last_xy[1])
                speeds.append(dist / dt)
                dts.append(dt)
            else:
                speeds.append(0.0)
                dts.append(0.0)
        last_t, last_xy = t, (x, y)
    if speeds:
        speeds.insert(0, speeds[0])  # align speeds length to pts for convenience
    else:
        speeds = [0.0 for _ in pts_px]
    if dts:
        dts.insert(0, dts[0])
    else:
        dts = [0.0 for _ in pts_px]
    return pts_px, speeds, dts


def render_scene(svg_path: Path) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cairosvg.svg2png(url=str(svg_path), write_to=tmp.name)
        img = Image.open(tmp.name).convert("RGBA")
    return img


def interp_color(val: float, vmin: float, vmax: float) -> Tuple[int, int, int, int]:
    # map val to gradient red->yellow->green
    if vmax <= vmin:
        return (0, 255, 0, 200)
    t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    if t < 0.5:
        # red -> yellow
        tt = t * 2
        return (255, int(255 * tt), 0, 200)
    else:
        # yellow -> green
        tt = (t - 0.5) * 2
        return (int(255 * (1 - tt)), 255, 0, 200)


def draw_dashed_line(draw: ImageDraw.ImageDraw, p1, p2, color, dash_len=6, gap_len=4, width=3):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    ux, uy = dx / dist, dy / dist
    x, y = p1
    traveled = 0.0
    while traveled < dist:
        seg = min(dash_len, dist - traveled)
        x2 = x + ux * seg
        y2 = y + uy * seg
        draw.line([(x, y), (x2, y2)], fill=color, width=width)
        traveled += dash_len + gap_len
        x += ux * (dash_len + gap_len)
        y += uy * (dash_len + gap_len)


def draw_traj(
    img: Image.Image,
    pts: List[Tuple[float, float]],
    speeds: List[float],
    dts: List[float],
    arrow_every: int = 3,
    dash_threshold: float = 5.0,
):
    draw = ImageDraw.Draw(img, "RGBA")
    if len(pts) >= 2:
        vmin, vmax = min(speeds), max(speeds)
        for i in range(len(pts) - 1):
            c = interp_color(speeds[i + 1], vmin, vmax)
            p1, p2 = pts[i], pts[i + 1]
            if dts[i + 1] > dash_threshold:
                draw_dashed_line(draw, p1, p2, c, width=3)
            else:
                draw.line([p1, p2], fill=c, width=3)
            if (i % arrow_every) == 0:
                # draw arrowhead at p2
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                norm = math.hypot(dx, dy) or 1.0
                ux, uy = dx / norm, dy / norm
                size = 8
                left = (p2[0] - ux * size + uy * size * 0.5, p2[1] - uy * size - ux * size * 0.5)
                right = (p2[0] - ux * size - uy * size * 0.5, p2[1] - uy * size + ux * size * 0.5)
                draw.polygon([p2, left, right], fill=c)
    if pts:
        r = 4
        draw.ellipse(
            (pts[0][0] - r, pts[0][1] - r, pts[0][0] + r, pts[0][1] + r),
            fill=(0, 0, 0, 200),
            outline=(0, 0, 0),
        )
        draw.ellipse(
            (pts[-1][0] - r, pts[-1][1] - r, pts[-1][0] + r, pts[-1][1] + r),
            fill=(0, 0, 255, 200),
            outline=(0, 0, 0),
        )


def composite_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def draw_legend(img: Image.Image, vmin: float, vmax: float):
    """Draw speed legend (m/s) with red->yellow->green gradient (top-left)."""
    draw = ImageDraw.Draw(img, "RGBA")
    bar_h = 200
    bar_w = 30
    x0 = 20
    y0 = 20
    # background box
    draw.rectangle([x0 - 6, y0 - 6, x0 + bar_w + 6, y0 + bar_h + 30], fill=(255, 255, 255, 230), outline=(0, 0, 0, 255))
    steps = bar_h
    for i in range(steps):
        t = i / (steps - 1)
        val = vmin + t * (vmax - vmin)
        c = interp_color(val, vmin, vmax)
        y = y0 + bar_h - 1 - i
        draw.line([(x0, y), (x0 + bar_w, y)], fill=c, width=1)
    font = ImageFont.load_default()
    draw.text((x0, y0 + bar_h + 5), "m/s", fill=(0, 0, 0), font=font)
    draw.text((x0 + bar_w + 8, y0 - 4), f"{vmax:.2f}", fill=(0, 0, 0), font=font)
    draw.text((x0 + bar_w + 8, y0 + bar_h - 10), f"{vmin:.2f}", fill=(0, 0, 0), font=font)


def main():
    parser = argparse.ArgumentParser(description="Sample random trajectory segments and visualize each individually.")
    parser.add_argument("--output", default="samples", help="output directory")
    parser.add_argument("--per-scene", type=int, default=100, help="number of segments per scene to sample")
    parser.add_argument("--gap-minutes", type=float, default=5.0, help="time gap (minutes) to split trajectories")
    parser.add_argument("--min-points", type=int, default=50, help="discard segments shorter than this many points")
    parser.add_argument("--dash-threshold", type=float, default=5.0, help="dt seconds above which line is dashed")
    args = parser.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(exist_ok=True)

    for scene, cfg in SCENES.items():
        segments = load_all_segments(cfg["data"], gap_minutes=args.gap_minutes)
        segments = [s for s in segments if len(s["pts"]) >= args.min_points]
        random.shuffle(segments)
        chosen = segments[: args.per_scene]
        scene_dir = out_root / scene
        scene_dir.mkdir(exist_ok=True)

        print(f"{scene}: segments total={len(segments)}, sampled={len(chosen)}")
        base_img = render_scene(cfg["svg"])

        for i, seg in enumerate(chosen):
            pts_px, speeds, dts = convert_pts(seg["pts"], origin_px=cfg["origin_px"], scale=cfg["scale_px_per_m"])
            img = base_img.copy()
            draw_traj(img, pts_px, speeds, dts, dash_threshold=args.dash_threshold)
            if speeds:
                draw_legend(img, min(speeds), max(speeds))
            out_path = scene_dir / f"sample_{i:03d}.png"
            composite_rgb(img).save(out_path)

        print(f"  saved to {scene_dir}")


if __name__ == "__main__":
    main()
