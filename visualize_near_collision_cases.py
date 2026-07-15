
import argparse

import json

import pickle

from pathlib import Path



import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import numpy as np

import torch



from scenario_map import create_scenario_map_3channel





def to_numpy(x):

    if x is None:

        return None

    if isinstance(x, torch.Tensor):

        return x.detach().cpu().numpy()

    return np.asarray(x)





def load_generated_pickle(path):

    with open(path, "rb") as f:

        obj = pickle.load(f)



    if isinstance(obj, dict):

        return obj



    if not isinstance(obj, (list, tuple)) or len(obj) < 7:

        raise ValueError("Unsupported generated output format.")



    return {

        "samples": obj[0],

        "target": obj[1],

        "evalpoint": obj[2],

        "observed_point": obj[3],

        "observed_time": obj[4],

        "scaler": obj[5],

        "mean_scaler": obj[6],

        "context": obj[7] if len(obj) > 7 and isinstance(obj[7], dict) else {},

    }





def fix_map_shape(scen_map):

    if scen_map is None:

        return None

    m = to_numpy(scen_map)

    if m.ndim == 3 and m.shape[0] in (3, 5) and m.shape[-1] not in (3, 5):

        m = np.transpose(m[:3], (1, 2, 0))

    if m.ndim == 3 and m.shape[-1] > 3:

        m = m[..., :3]

    return m





def load_scenario_map_background(scenario, data_folder="./data/simulation_data/", scale=10):

    path = Path(data_folder) / f"Scenario{scenario}" / "scenario_map.json"

    with open(path, "r") as f:

        scenario_def = json.load(f)

    return create_scenario_map_3channel(scenario_def, scale=scale)





def draw_map(ax, scen_map):

    scen_map = fix_map_shape(scen_map)

    if scen_map is None:

        ax.set_facecolor("white")

        return



    m = scen_map.astype(np.float32)

    if m.max() > 1.0:

        m = m / 255.0



    h, w = m.shape[:2]

    rgb = np.zeros((h, w, 3), dtype=np.float32)

    rgb[m[:, :, 0] > 0.5] = [0.0, 0.0, 1.0]

    rgb[m[:, :, 1] > 0.5] = [0.0, 1.0, 0.0]

    rgb[m[:, :, 2] > 0.5] = [1.0, 0.0, 0.0]

    ax.imshow(rgb, extent=[0, 1, 1, 0], aspect="auto")





def collision_mask_from_map(scen_map):

    scen_map = fix_map_shape(scen_map)

    if scen_map is None:

        return None

    m = scen_map.astype(np.float32)

    if m.max() > 1.0:

        m = m / 255.0

    return (m[..., 0] > 0.5) | (m[..., 1] > 0.5)





def add_direction_arrow(ax, points, valid, color, lw=2.5, mutation_scale=18, zorder=20):

    idx = np.where(valid)[0]

    if len(idx) < 2:

        return



    end_idx = idx[-1]

    start_idx = idx[-2]

    for cand in idx[-2::-1]:

        if np.linalg.norm(points[end_idx] - points[cand]) > 0.006:

            start_idx = cand

            break



    p0 = points[start_idx]

    p1 = points[end_idx]

    if np.linalg.norm(p1 - p0) <= 1e-8:

        return



    ax.annotate(

        "",

        xy=(p1[0], p1[1]),

        xytext=(p0[0], p0[1]),

        arrowprops=dict(

            arrowstyle="-|>",

            color=color,

            lw=lw,

            mutation_scale=mutation_scale,

            shrinkA=0,

            shrinkB=0,

        ),

        zorder=zorder,

    )





def local_obstacle_score(points, valid, scen_map, radius_px=12):

    mask = collision_mask_from_map(scen_map)

    if mask is None or valid.sum() == 0:

        return float("inf")



    h, w = mask.shape

    pts = points[valid]

    if len(pts) == 0:

        return float("inf")



    best = float("inf")

    for p in pts:

        x = int(np.clip(p[0] * w, 0, w - 1))

        y = int(np.clip(p[1] * h, 0, h - 1))



        y0 = max(0, y - radius_px)

        y1 = min(h, y + radius_px + 1)

        x0 = max(0, x - radius_px)

        x1 = min(w, x + radius_px + 1)



        local = mask[y0:y1, x0:x1]

        if local.any():

            ly, lx = np.where(local)

            dy = ly + y0 - y

            dx = lx + x0 - x

            d = np.sqrt(dx * dx + dy * dy).min()

            best = min(best, float(d) / max(h, w))



    return best





def select_validity_aware_index(samples, evalpoint, scen_map=None):

    valid = evalpoint[..., 0] > 0

    median = np.median(samples, axis=0)

    invalid_counts = np.zeros(samples.shape[0], dtype=np.float64)



    mask = collision_mask_from_map(scen_map)

    if mask is not None:

        h, w = mask.shape

        for i, sample in enumerate(samples):

            x_float = sample[:, 0] * w

            y_float = sample[:, 1] * h

            oob = (x_float < 0) | (x_float >= w) | (y_float < 0) | (y_float >= h)

            x = np.clip(x_float.astype(np.int64), 0, w - 1)

            y = np.clip(y_float.astype(np.int64), 0, h - 1)

            invalid_counts[i] += np.logical_or(mask[y, x], oob)[valid].sum()



    median_dist = []

    for s in samples:

        if valid.sum() > 0:

            median_dist.append(float(np.linalg.norm((s - median)[valid], axis=-1).mean()))

        else:

            median_dist.append(0.0)



    score = invalid_counts * 1e6 + np.asarray(median_dist)

    return int(np.argmin(score))





def obstacle_case_score(data, case_id, fallback_scen_map):

    samples = to_numpy(data["samples"])

    target = to_numpy(data["target"])

    evalpoint = to_numpy(data["evalpoint"])

    context = data.get("context", {}) or {}

    scen_map = to_numpy(context.get("scen_map"))



    case_map = fix_map_shape(scen_map[case_id]) if scen_map is not None else fallback_scen_map

    valid = evalpoint[case_id, :, 0] > 0



    ns = min(30, samples.shape[1])

    selected = select_validity_aware_index(samples[case_id, :ns], evalpoint[case_id], scen_map=case_map)



    gt_score = local_obstacle_score(target[case_id], valid, case_map, radius_px=12)

    pred_score = local_obstacle_score(samples[case_id, selected], valid, case_map, radius_px=12)



    return min(gt_score, pred_score)





def choose_cases(data, num_cases, fallback_scen_map):

    b = to_numpy(data["target"]).shape[0]

    scores = []

    for i in range(b):

        score = obstacle_case_score(data, i, fallback_scen_map)

        scores.append((score, i))



    finite = [(score, i) for score, i in scores if np.isfinite(score)]

    if finite:

        finite.sort(key=lambda x: x[0])

        return [int(i) for _, i in finite[:min(num_cases, len(finite))]]



    return list(range(min(num_cases, b)))





def plot_case(data, case_id, out_path, fallback_scen_map, max_samples=30):

    samples = to_numpy(data["samples"])

    target = to_numpy(data["target"])

    evalpoint = to_numpy(data["evalpoint"])

    context = data.get("context", {}) or {}

    scen_map = to_numpy(context.get("scen_map"))



    case_map = fix_map_shape(scen_map[case_id]) if scen_map is not None else fallback_scen_map



    s = samples[case_id]

    t = target[case_id]

    e = evalpoint[case_id]

    valid_future = e[:, 0] > 0



    if case_map is not None:

        h, w = case_map.shape[:2]

        aspect_ratio = w / h

        if aspect_ratio > 2.0:

            fig, ax = plt.subplots(figsize=(22, 22 / aspect_ratio + 1))

        else:

            fig, ax = plt.subplots(figsize=(8, 8))

    else:

        fig, ax = plt.subplots(figsize=(8, 8))



    draw_map(ax, case_map)



    if valid_future.sum() > 1:

        ax.plot(t[valid_future, 0], t[valid_future, 1], color="tab:green", lw=3.0, label="ground truth", zorder=16)

        add_direction_arrow(ax, t, valid_future, color="tab:green", lw=3.0, mutation_scale=20, zorder=17)



    ns = min(max_samples, s.shape[0])

    selected = select_validity_aware_index(s[:ns], e, scen_map=case_map)

    pred = s[selected]



    if valid_future.sum() > 1:

        ax.plot(pred[valid_future, 0], pred[valid_future, 1], color="tab:red", lw=3.2, label="selected prediction", zorder=18)

        add_direction_arrow(ax, pred, valid_future, color="tab:red", lw=3.2, mutation_scale=22, zorder=19)



    score = obstacle_case_score(data, case_id, fallback_scen_map)

    ax.set_title(f"case {case_id} | selected={selected} | obstacle_score={score:.4f}")

    ax.set_xlim(0, 1)

    ax.set_ylim(1, 0)

    ax.set_aspect("auto")

    ax.grid(False)



    handles, labels = ax.get_legend_handles_labels()

    by_label = dict(zip(labels, handles))

    wall_patch = mpatches.Patch(color="lime", label="wall")

    obs_patch = mpatches.Patch(color="blue", label="obstacle")



    custom_handles = [

        by_label.get("ground truth"),

        by_label.get("selected prediction"),

        wall_patch,

        obs_patch,

    ]



    ax.legend(

        handles=[h for h in custom_handles if h is not None],

        loc="upper left",

        bbox_to_anchor=(1.01, 1),

        fontsize="small",

    )



    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")

    plt.close(fig)





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--pickle", required=True)

    parser.add_argument("--outdir", required=True)

    parser.add_argument("--scenario", required=True)

    parser.add_argument("--num_cases", type=int, default=5)

    parser.add_argument("--max_neighbors", type=int, default=0)

    parser.add_argument("--max_samples", type=int, default=30)

    parser.add_argument("--data_folder", default="./data/simulation_data/")

    parser.add_argument("--scen_map_scale", type=int, default=10)

    args = parser.parse_args()



    data = load_generated_pickle(args.pickle)

    fallback_scen_map = load_scenario_map_background(

        args.scenario,

        data_folder=args.data_folder,

        scale=args.scen_map_scale,

    )



    outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)



    case_ids = choose_cases(data, args.num_cases, fallback_scen_map)

    print("selected cases:", case_ids)



    for cid in case_ids:

        out_path = outdir / f"near_case_{cid:04d}.png"

        plot_case(

            data,

            cid,

            out_path,

            fallback_scen_map=fallback_scen_map,

            max_samples=args.max_samples,

        )

        print(f"wrote {out_path}")





if __name__ == "__main__":

    main()

