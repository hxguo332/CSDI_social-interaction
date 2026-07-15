#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


DEFAULT_SCENARIOS = ["1-1", "2-1", "2-2", "2-3", "3-1", "3-2", "4-1"]


def load_paths(data_folder: str, scenario: str) -> Tuple[str, str]:
    """Return (path_sim, path_info) for a scenario, matching dataset_simulation._load_data."""
    if scenario == "4-1":
        path_sim = os.path.join(data_folder, "Scenario4-1/outputdata-1.6.2021/simulationLog_clean.csv")
        path_info = os.path.join(data_folder, "Scenario4-1/outputdata-1.6.2021/outputPersonInfo_clean.csv")
    else:
        path_sim = os.path.join(
            data_folder,
            f"Scenario{scenario}/output data-2.15(yamada@vri)/simulationLog_clean.csv",
        )
        path_info = os.path.join(
            data_folder,
            f"Scenario{scenario}/output data-2.15(yamada@vri)/outputPersonInfo_clean.csv",
        )
    return path_sim, path_info


def filter_sim_info(person_info: pd.DataFrame, sim_info: pd.DataFrame, time_limit: float = 200) -> pd.DataFrame:
    """Apply the same filtering as Simulation_Dataset._filter_data."""
    # Remove standing persons (PersonType == 3)
    person_ids_to_remove = person_info[person_info["PersonType"] == 3]["PersonID"]
    filtered = sim_info[~sim_info["personID"].isin(person_ids_to_remove)]
    # Keep rows where time <= time_limit
    filtered = filtered[filtered["time"] <= time_limit]
    return filtered


def count_tracks_for_scenario(data_folder: str, scenario: str, time_limit: float) -> int:
    path_sim, path_info = load_paths(data_folder, scenario)
    if not (os.path.exists(path_sim) and os.path.exists(path_info)):
        raise FileNotFoundError(f"Missing files for scenario {scenario}:\n  {path_sim}\n  {path_info}")
    person_info = pd.read_csv(path_info)
    sim_info = pd.read_csv(path_sim)
    sim_info = filter_sim_info(person_info, sim_info, time_limit=time_limit)
    # Count unique person tracks (before splitting into fixed length)
    return int(sim_info["personID"].nunique())


def main():
    parser = argparse.ArgumentParser(description="Count tracks per scenario before fixed-length splitting.")
    parser.add_argument(
        "--data-folder",
        type=str,
        default="./data/simulation_data/",
        help="Root folder for simulation data (matches dataset_simulation).",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated list of scenarios to include. Default matches dataset.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=200.0,
        help="Time limit filter (same as dataset_simulation._filter_data).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable text.",
    )

    args = parser.parse_args()
    scenarios: List[str] = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    counts = {}
    total = 0
    for sc in scenarios:
        try:
            n = count_tracks_for_scenario(args.data_folder, sc, args.time_limit)
        except Exception as e:
            raise
        counts[sc] = n
        total += n

    if args.json:
        out = {"total": total, "per_scenario": counts}
        print(json.dumps(out, indent=2))
    else:
        print("Tracks before fixed-length split:")
        for sc in scenarios:
            print(f"  Scenario {sc}: {counts[sc]}")
        print(f"Total: {total}")


if __name__ == "__main__":
    main()

