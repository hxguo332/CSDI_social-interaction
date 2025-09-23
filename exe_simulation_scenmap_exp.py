import argparse
import datetime
import json
import os
import yaml
import torch

from main_model import CSDI_SimulationScenmap
from dataset_augmented_simulation import (
    get_augmented_dataloader,
    get_nonaugmented_dataloader_with_scenario_batches,
)
from utils import train, evaluate, build_preproc_tag_from_split_cfg


def save_config(config, args):
    """Save configuration and args to a timestamped folder and return its path."""
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = os.path.join("./save", f"simulation_{current_time}")
    print("model folder:", foldername)
    os.makedirs(foldername, exist_ok=True)

    merged = dict(config)
    merged["args"] = vars(args)
    with open(os.path.join(foldername, "config.json"), "w") as f:
        json.dump(merged, f, indent=4)
    return foldername


def build_loader_from_cfg(split_name, cfg, common_kwargs):
    """
    Build dataloader (and optional batch sampler) for a split based on config.

    Returns: (loader, batch_sampler_or_None)
    """
    dl_type = cfg.get("dataloader", "augmented")

    if dl_type == "augmented":
        # Expect augmentation options per split
        aug_mode = cfg.get("augmentation_mode", "crop_and_resize")
        resize_options = cfg.get("resize_options", [[512, 512]])
        loader, batch_sampler = get_augmented_dataloader(
            part=split_name,
            augmentation_mode=aug_mode,
            resize_options=resize_options,
            **common_kwargs,
        )
        return loader, batch_sampler

    elif dl_type in ("nonaugmented", "nonaugmented_scenario_batch", "scenario_batch"):
        loader = get_nonaugmented_dataloader_with_scenario_batches(
            part=split_name,
            **common_kwargs,
        )
        return loader, None

    else:
        raise ValueError(f"Unknown dataloader type for split '{split_name}': {dl_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSDI scenemap experiments")
    parser.add_argument("--config", type=str, default="exp_scenmap_aug.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--testmissingratio", type=float, default=0.1)
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--data_length", type=int, default=100)
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="normalized",
        choices=["normalized", "unnormalized"],
        help="Sampling mode for evaluation",
    )

    args = parser.parse_args()
    print(args)

    # Load YAML config
    cfg_path = os.path.join("config", args.config)
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # Thread common model options
    config["model"]["test_missing_ratio"] = args.testmissingratio
    print(json.dumps(config, indent=4))

    # Prepare output folder and persist config+args
    foldername = args.modelfolder or save_config(config, args)

    # Common dataloader kwargs
    scenarios = None if config.get("dataset", {}).get("scenarios") in (None, "null") else config["dataset"]["scenarios"]
    common_loader_kwargs = dict(
        scenarios=scenarios,
        data_length=args.data_length,
        seed=args.seed,
        batch_size=config["train"]["batch_size"],  # will be overridden below per split
        zero_based_position=True,
        load_scenario_map=True,
        debug=False,
    )

    # Train loader
    common_loader_kwargs["batch_size"] = config["train"]["batch_size"]
    train_loader, batch_sampler_train = build_loader_from_cfg("train", config["train"], common_loader_kwargs)

    # Valid loader
    valid_loader = None
    batch_sampler_valid = None
    if "valid" in config and config["valid"] is not None:
        common_loader_kwargs["batch_size"] = config["valid"].get("batch_size", config["train"]["batch_size"])
        valid_loader, batch_sampler_valid = build_loader_from_cfg("valid", config["valid"], common_loader_kwargs)

    # Test loader
    common_loader_kwargs["batch_size"] = config.get("test", {}).get("batch_size", config["train"]["batch_size"])
    test_loader, _ = build_loader_from_cfg("test", config.get("test", {}), common_loader_kwargs)

    # Build model
    model = CSDI_SimulationScenmap(config, args.device).to(args.device)

    # Train or load
    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
            batch_sampler_train=batch_sampler_train,
            batch_sampler_valid=batch_sampler_valid,
        )
    else:
        model.load_state_dict(torch.load(os.path.join(foldername, "model.pth")))

    # Evaluation
    sample_mode = args.sample_mode
    eval_collision = config.get("test", {}).get("eval_collision", False)
    # Derive a preprocessing tag from the test split config for filename suffixing
    preproc_tag = build_preproc_tag_from_split_cfg(config.get("test", {}), fallback_name=None)
    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=1,
        foldername=foldername,
        mode=sample_mode,
        eval_collision=eval_collision,
        file_tag=preproc_tag,
    )
