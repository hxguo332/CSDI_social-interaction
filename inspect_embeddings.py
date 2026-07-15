import argparse
import os
import json
import yaml
import torch
from main_model import CSDI_Simulation
from dataset_simulation import get_dataloader


def running_stats_init():
    return {
        "count": 0,
        "sum": 0.0,
        "sumsq": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
    }


def running_stats_update(stats, tensor):
    # Flatten to 1D for simplicity
    x = tensor.detach().reshape(-1).float()
    stats["count"] += x.numel()
    stats["sum"] += x.sum().item()
    stats["sumsq"] += (x * x).sum().item()
    stats["min"] = min(stats["min"], float(x.min().item()))
    stats["max"] = max(stats["max"], float(x.max().item()))


def running_stats_finalize(stats):
    if stats["count"] == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    mean = stats["sum"] / stats["count"]
    var = max(0.0, stats["sumsq"] / stats["count"] - mean * mean)
    std = var ** 0.5
    return {
        "count": int(stats["count"]),
        "min": stats["min"],
        "max": stats["max"],
        "mean": mean,
        "std": std,
    }


def inspect_embeddings(config_path: str, model_folder: str, device: str, data_length: int, seed: int):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Build dataloader matching training loader
    batch_size = config["train"]["batch_size"]
    train_loader, _, _ = get_dataloader(
        data_length=data_length,
        seed=seed,
        batch_size=batch_size,
        zero_based_position=True,
    )

    # Build model and load weights
    model = CSDI_Simulation(config, device).to(device)
    ckpt = os.path.join(model_folder, "model.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Stats holders
    time_stats = running_stats_init()
    # feature embedding weights are global; just compute once
    feat_weight = model.embed_layer.weight.detach().to(device)
    feat_stats = running_stats_init()
    running_stats_update(feat_stats, feat_weight)

    # Also collect side_info stats (optional diagnostic)
    side_stats = running_stats_init()

    with torch.no_grad():
        for batch in train_loader:
            # time_embed: shape (B, L, emb_time_dim)
            observed_tp = batch["timepoints"].to(device).float()
            time_embed = model.time_embedding(observed_tp, model.emb_time_dim)
            running_stats_update(time_stats, time_embed)

            # For side_info stats (constructed as in training forward for Simulation)
            observed_mask = batch["observed_mask"].to(device).float().permute(0, 2, 1)
            gt_mask = batch["gt_mask"].to(device).float().permute(0, 2, 1)
            cond_mask = model.get_test_pattern_mask(observed_mask, gt_mask)
            side_info = model.get_side_info(observed_tp, cond_mask)
            running_stats_update(side_stats, side_info)

    results = {
        "time_embed": running_stats_finalize(time_stats),
        "feature_embed_weight": running_stats_finalize(feat_stats),
        "side_info": running_stats_finalize(side_stats),
        "meta": {
            "emb_time_dim": int(model.emb_time_dim),
            "emb_feature_dim": int(model.emb_feature_dim),
            "target_dim": int(model.target_dim),
        },
    }

    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Inspect embedding/side_info statistics after training")
    parser.add_argument("--config", type=str, default="config/base.yaml")
    parser.add_argument("--model_folder", type=str, required=True, help="Path to folder containing model.pth (e.g., ./save/simulation_YYYYMMDD_HHMMSS)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    inspect_embeddings(args.config, args.model_folder, args.device, args.data_length, args.seed)


if __name__ == "__main__":
    main()

