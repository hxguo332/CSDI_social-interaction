#!/usr/bin/env python3
"""
Print raw scenario map sizes (no augmentation / no resize).

For each ScenarioX-Y under data/simulation_data, this script reads
scenario_map.json, computes the world extents, and reports the pixel
dimensions at a chosen base scale (default 10). Optionally verifies
the shape by constructing the 3-channel map.

Usage:
  python tools/print_scenemap_sizes.py \
    [--root ./data/simulation_data] [--scale 10] [--scenarios 1-1,2-1,...] [--verify]
"""

import argparse
import json
import os
from typing import List


def discover_scenarios(root: str) -> List[str]:
    scenarios = []
    if not os.path.isdir(root):
        return scenarios
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        if not name.startswith("Scenario"):
            continue
        scen_id = name[len("Scenario"):]
        if os.path.isfile(os.path.join(d, "scenario_map.json")):
            scenarios.append(scen_id)
    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Print raw (unresized) scenemap sizes")
    parser.add_argument("--root", type=str, default="./data/simulation_data", help="Root folder containing Scenario*/")
    parser.add_argument("--scale", type=int, default=10, help="Base pixels-per-world-unit scale used to construct maps")
    parser.add_argument("--scenarios", type=str, default="", help="Comma-separated list like '1-1,2-1'; defaults to auto-discovery")
    parser.add_argument("--verify", action="store_true", help="Construct the 3-channel map to verify the reported shape")
    args = parser.parse_args()

    if args.scenarios.strip():
        scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    else:
        scenarios = discover_scenarios(args.root)

    if not scenarios:
        print("No scenarios found.")
        return

    # Optional import only if verification requested
    create_fn = None
    if args.verify:
        try:
            from scenario_map import create_scenario_map_3channel  # type: ignore
            create_fn = create_scenario_map_3channel
        except Exception as e:
            print(f"Warning: could not import scenario_map.create_scenario_map_3channel: {e}")

    print(f"Root: {args.root}")
    print(f"Base scale (px per world unit): {args.scale}")
    print("")

    for scen in scenarios:
        scen_dir = os.path.join(args.root, f"Scenario{scen}")
        json_path = os.path.join(scen_dir, "scenario_map.json")
        if not os.path.isfile(json_path):
            print(f"Scenario {scen}: missing {json_path}")
            continue
        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Scenario {scen}: failed to read JSON: {e}")
            continue

        pos = meta.get("position", {})
        p1 = pos.get("p1")
        p2 = pos.get("p2")
        if not (isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple)) and len(p1) == 2 and len(p2) == 2):
            print(f"Scenario {scen}: invalid position format in JSON")
            continue

        width_world = abs(float(p2[0]) - float(p1[0]))
        height_world = abs(float(p2[1]) - float(p1[1]))
        W_px = int(round(width_world * args.scale))
        H_px = int(round(height_world * args.scale))

        line = (
            f"Scenario {scen}: world W={width_world:g}, H={height_world:g} -> "
            f"pixels HxW={H_px}x{W_px}"
        )

        if create_fn is not None:
            try:
                grid = create_fn(meta, scale=args.scale, draw_standing_person=False)
                line += f" | verify grid.shape={tuple(grid.shape)}"
            except Exception as e:
                line += f" | verify failed: {e}"

        print(line)


if __name__ == "__main__":
    main()

