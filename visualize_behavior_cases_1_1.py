import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import json

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


def draw_map(ax, scen_map):
    if scen_map is None:
        ax.set_facecolor("white")
        return
    
    # 1. 数据标准化
    m = scen_map.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0

    # 2. 创建一个新的 RGB 画布 [H, W, 3]
    h, w = m.shape[:2]
    display_rgb = np.zeros((h, w, 3), dtype=np.float32)

    # 3. 根据你的通道定义精准分配颜色：
    
    # 通道 0：障碍物 (Obstacles) -> 显示为蓝色 [0, 0, 1]
    display_rgb[m[:, :, 0] > 0.5] = [0, 0, 1]
    
    # 通道 1：墙壁/禁止区域 (Walls/Forbidden) -> 显示为绿色 [0, 1, 0]
    display_rgb[m[:, :, 1] > 0.5] = [0, 1, 0]
    
    # 通道 2：出入口 (Entrances) -> 显示为红色 [1, 0, 0]
    display_rgb[m[:, :, 2] > 0.5] = [1, 0, 0]

    # 4. 绘图
    # 保持 aspect='auto' 以配合你之前设置好的 base_width=22 铺满全屏
    ax.imshow(display_rgb, extent=[0, 1, 1, 0], aspect='auto')


def add_direction_arrow(ax, points, valid, color, lw=2.4, mutation_scale=18):
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return

    end_idx = idx[-1]
    start_idx = idx[-2]
    # Walk backward until the arrow segment is long enough to be visible.
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
        zorder=20,
    )


def pairwise_ade(a, b, valid):
    if valid.sum() == 0:
        return 0.0
    return float(np.linalg.norm(a[valid] - b[valid], axis=-1).mean())


def cluster_samples(samples, valid_future, threshold=0.03):
    n = samples.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if pairwise_ade(samples[i], samples[j], valid_future) <= threshold:
                union(i, j)
    roots = [find(i) for i in range(n)]
    uniq = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.asarray([uniq[r] for r in roots], dtype=np.int64)


def select_validity_aware_index(samples, evalpoint, scen_map=None, neighbor_data=None, neighbor_mask=None, social_margin=0.003):
    valid = evalpoint[..., 0] > 0
    median = np.median(samples, axis=0)
    invalid_counts = np.zeros(samples.shape[0], dtype=np.float64)

    if scen_map is not None:
        m = scen_map.astype(np.float32)
        if m.max() <= 1.0:
            collision_mask = (m[..., 0] > 0.5) | (m[..., 1] > 0.5)
        else:
            collision_mask = (m[..., 0] > 0) | (m[..., 1] > 0)
        h, w = collision_mask.shape
        for i, sample in enumerate(samples):
            x_float = sample[:, 0] * w
            y_float = sample[:, 1] * h
            oob = (x_float < 0) | (x_float >= w) | (y_float < 0) | (y_float >= h)
            x = np.clip(x_float.astype(np.int64), 0, w - 1)
            y = np.clip(y_float.astype(np.int64), 0, h - 1)
            invalid_counts[i] += np.logical_or(collision_mask[y, x], oob)[valid].sum()

    if neighbor_data is not None and neighbor_mask is not None:
        for i, sample in enumerate(samples):
            dist = np.linalg.norm(sample[None, :, :] - neighbor_data, axis=-1)
            social_invalid = ((dist < social_margin) & (neighbor_mask > 0)).any(axis=0)
            invalid_counts[i] += social_invalid[valid].sum()

    median_dist = np.asarray([pairwise_ade(s, median, valid) for s in samples])
    score = invalid_counts * 1e6 + median_dist
    return int(np.argmin(score))


def load_scenario_map_background(scenario, data_folder="./data/simulation_data/", scale=10):
    path = Path(data_folder) / f"Scenario{scenario}" / "scenario_map.json"
    with open(path, "r") as f:
        scenario_def = json.load(f)
    return create_scenario_map_3channel(scenario_def, scale=scale)


def plot_case(data, case_id, out_path, max_samples=30, mode_threshold=0.03, fallback_scen_map=None):
    samples = to_numpy(data["samples"])          # [B,S,L,2]
    target = to_numpy(data["target"])            # [B,L,2]
    evalpoint = to_numpy(data["evalpoint"])      # [B,L,2]
    observed = to_numpy(data["observed_point"])  # [B,L,2]
    context = data.get("context", {}) or {}
    neighbor_data = to_numpy(context.get("neighbor_data"))
    neighbor_mask = to_numpy(context.get("neighbor_mask"))
    scen_map = to_numpy(context.get("scen_map"))
    conflict = to_numpy(context.get("conflict_features"))

    s = samples[case_id]
    t = target[case_id]
    e = evalpoint[case_id]
    o = observed[case_id]
    valid_future = e[..., 0] > 0
    valid_obs = ~valid_future

    # 1. 确定地图对象
    case_map = scen_map[case_id] if scen_map is not None else fallback_scen_map
    
    # 2. 根据地图真实的像素比例来决定画布形状
    if case_map is not None:
        h, w = case_map.shape[:2]
        aspect_ratio = w / h  # 计算宽/高比
        
        # 如果宽度是高度的 2 倍以上，判定为长条场景（如 1-1）
        if aspect_ratio > 2.0:
            base_width = 22  # 继续加大宽度
            dynamic_height = base_width / aspect_ratio
        # 稍微增加一点高度给标题留空间
            fig, ax = plt.subplots(figsize=(base_width, dynamic_height + 1))
        else:
            # 普通比例场景（如 4-1）
            fig, ax = plt.subplots(figsize=(8, 8))
    else:
        # 无地图情况默认正方形
        fig, ax = plt.subplots(figsize=(8, 8))
        
    draw_map(ax, case_map)

    if neighbor_data is not None and neighbor_mask is not None:
        nd = neighbor_data[case_id]
        nm = neighbor_mask[case_id]
        for n in range(nd.shape[0]):
            valid_n = nm[n] > 0
            if valid_n.sum() > 1:
                ax.plot(
                    nd[n, valid_n, 0],
                    nd[n, valid_n, 1],
                    color="orange",
                    lw=1.4,
                    alpha=0.75,
                    label="neighbors" if n == 0 else None,
                )
                add_direction_arrow(ax, nd[n], valid_n, color="orange", lw=1.5, mutation_scale=13)

    if valid_obs.sum() > 1:
        ax.plot(o[valid_obs, 0], o[valid_obs, 1], color="lightskyblue", lw=2.4, label="ego observed")
        add_direction_arrow(ax, o, valid_obs, color="lightskyblue", lw=2.6, mutation_scale=20)
    if valid_future.sum() > 1:
        ax.plot(t[valid_future, 0], t[valid_future, 1], color="tab:green", lw=2.4, label="ground truth")
        add_direction_arrow(ax, t, valid_future, color="tab:green", lw=2.6, mutation_scale=20)

    ns = min(max_samples, s.shape[0])
    labels = cluster_samples(s[:ns], valid_future, threshold=mode_threshold)
    n_modes = max(int(labels.max()) + 1, 1) if len(labels) else 1
    for i in range(ns):
        ax.plot(
            s[i, valid_future, 0],
            s[i, valid_future, 1],
            color="lightskyblue",
            lw=1.0,
            alpha=0.55,
            label="generated samples" if i == 0 else None,
        )
        add_direction_arrow(ax, s[i], valid_future, color="lightskyblue", lw=1.0, mutation_scale=10)

    nd_case = neighbor_data[case_id] if neighbor_data is not None else None
    nm_case = neighbor_mask[case_id] if neighbor_mask is not None else None
    selected = select_validity_aware_index(
        s[:ns],
        e,
        scen_map=case_map,
        neighbor_data=nd_case,
        neighbor_mask=nm_case,
    )
    ax.plot(
        s[selected, valid_future, 0],
        s[selected, valid_future, 1],
        color="tab:red",
        lw=3.0,
        alpha=0.95,
        label="selected",
    )
    add_direction_arrow(ax, s[selected], valid_future, color="tab:red", lw=3.2, mutation_scale=22)

    title = f"case {case_id}"
    if conflict is not None:
        c = conflict[case_id]
        title += f" | d={c[0]:.3f}, TTC={c[1]:.3f}, heading={c[2]:.3f}"
    ax.set_title(title + f" | modes={n_modes}, selected={selected}")
    ax.set_ylim(1, 0)
    ax.set_xlim(0, 1)
    ax.set_aspect('auto')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='small')
    ax.grid(False)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def choose_cases(data, num_cases=8, prefer_high_conflict=True):
    context = data.get("context", {}) or {}
    conflict = to_numpy(context.get("conflict_features"))
    b = to_numpy(data["samples"]).shape[0]
    if conflict is None or not prefer_high_conflict:
        return list(range(min(num_cases, b)))
    # Low min distance means higher conflict.
    order = np.argsort(conflict[:, 0])
    return [int(i) for i in order[: min(num_cases, b)]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", required=True, help="generated_outputs_nsample*.pk")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--case", type=int, default=None)
    parser.add_argument("--num_cases", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--mode_threshold", type=float, default=0.03)
    parser.add_argument("--scenario", default=None, help="Scenario id such as 3-1; used to reload map if pickle has no scen_map")
    parser.add_argument("--data_folder", default="./data/simulation_data/")
    parser.add_argument("--scen_map_scale", type=int, default=10)
    args = parser.parse_args()

    data = load_generated_pickle(args.pickle)
    outdir = Path(args.outdir) if args.outdir else Path(args.pickle).with_name("behavior_case_plots")
    fallback_scen_map = None
    if args.scenario is not None:
        fallback_scen_map = load_scenario_map_background(
            args.scenario,
            data_folder=args.data_folder,
            scale=args.scen_map_scale,
        )

    if args.case is not None:
        case_ids = [args.case]
    else:
        case_ids = choose_cases(data, num_cases=args.num_cases)

    for cid in case_ids:
        out_path = outdir / f"case_{cid:04d}.png"
        plot_case(
            data,
            cid,
            out_path,
            max_samples=args.max_samples,
            mode_threshold=args.mode_threshold,
            fallback_scen_map=fallback_scen_map,
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
