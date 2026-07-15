import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_SimulationScenmap, CSDI_SocialFusionScenmap
from dataset_simulation import get_dataloader, get_social_dataloader
from utils import train, evaluate


VARIANT_CHOICES = [
    "baseline",
    "task",
    "goal",
    "social",
    "fusion",
    "obs_loss",
    "social_loss",
    "full",
    "goal_full",
]


def _reset_ablation_flags(model_cfg):
    model_cfg["enable_goal_guidance"] = False
    model_cfg["enable_social_branch"] = False
    model_cfg["enable_game_fusion"] = False
    model_cfg["add_collision_loss"] = False
    model_cfg["add_social_collision_loss"] = False

    model_cfg.setdefault("scene_goal_channels", 5)
    model_cfg.setdefault("socialemb", 64)
    model_cfg.setdefault("social_hidden", 64)
    model_cfg.setdefault("social_hidden_dim", 64)
    model_cfg.setdefault("fusionemb", model_cfg.get("scenmapemb", 256))


def configure_variant_legacy(config, variant):
    """
    Legacy ablation order:
    Baseline
    + Goal Guidance
    + Social Branch
    + Game-Aware Fusion
    + Obstacle Collision Loss
    + Social Collision Loss
    """
    model_cfg = config.setdefault("model", {})
    _reset_ablation_flags(model_cfg)

    if variant == "baseline":
        return "base"

    if variant == "goal":
        model_cfg["enable_goal_guidance"] = True
        return "social"

    if variant == "social":
        model_cfg["enable_goal_guidance"] = True
        model_cfg["enable_social_branch"] = True
        return "social"

    if variant == "fusion":
        model_cfg["enable_goal_guidance"] = True
        model_cfg["enable_social_branch"] = True
        model_cfg["enable_game_fusion"] = True
        return "social"

    if variant == "obs_loss":
        model_cfg["enable_goal_guidance"] = True
        model_cfg["enable_social_branch"] = True
        model_cfg["enable_game_fusion"] = True
        model_cfg["add_collision_loss"] = True
        return "social"

    if variant == "full":
        model_cfg["enable_goal_guidance"] = True
        model_cfg["enable_social_branch"] = True
        model_cfg["enable_game_fusion"] = True
        model_cfg["add_collision_loss"] = True
        model_cfg["add_social_collision_loss"] = True
        return "social"

    raise ValueError(f"Unknown legacy model_variant: {variant}")


def configure_variant_reordered(config, variant):
    """
    Reordered ablation order:
    Baseline
    + Task Strategy
    + Social Branch
    + Game-Aware Fusion
    + Obstacle Collision Loss
    + Social Collision Loss
    + Goal Guidance
    """
    model_cfg = config.setdefault("model", {})
    _reset_ablation_flags(model_cfg)

    if variant == "baseline":
        return "base"

    if variant == "task":
        return "base"

    if variant == "social":
        model_cfg["enable_social_branch"] = True
        return "social"

    if variant == "fusion":
        model_cfg["enable_social_branch"] = True
        model_cfg["enable_game_fusion"] = True
        return "social"

    if variant == "obs_loss":
        model_cfg["enable_social_branch"] = True
        model_cfg["enable_game_fusion"] = True
        model_cfg["add_collision_loss"] = True
        return "social"

    if variant == "social_loss":
        model_cfg["enable_social_branch"] = True
        model_cfg["enable_game_fusion"] = True
        model_cfg["add_collision_loss"] = True
        model_cfg["add_social_collision_loss"] = True
        return "social"

    if variant == "goal_full":
        model_cfg["enable_goal_guidance"] = True
        model_cfg["enable_social_branch"] = True
        model_cfg["enable_game_fusion"] = True
        model_cfg["add_collision_loss"] = True
        model_cfg["add_social_collision_loss"] = True
        return "social"

    raise ValueError(f"Unknown reordered model_variant: {variant}")


def configure_variant(config, variant, ablation_order):
    if ablation_order == "legacy":
        return configure_variant_legacy(config, variant)

    if ablation_order == "reordered":
        return configure_variant_reordered(config, variant)

    raise ValueError(f"Unknown ablation_order: {ablation_order}")


def build_variant_tag(args):
    variant_tag = args.model_variant

    if args.force_random_strategy:
        variant_tag = f"random_{variant_tag}"

    if args.force_random_target_know_first:
        variant_tag = f"random_target_knowfirst_{variant_tag}"

    if args.ablation_order == "reordered":
        variant_tag = f"reordered_{variant_tag}"

    if args.ablation_order == "legacy":
        variant_tag = f"legacy_{variant_tag}"

    return variant_tag


def build_scenario_tag(config, args):
    scenarios = args.scenarios if args.scenarios is not None else config.get("dataset", {}).get("scenarios", None)

    if scenarios is None:
        return "all"

    if isinstance(scenarios, str):
        return scenarios

    if isinstance(scenarios, list):
        return "_".join(str(s) for s in scenarios)

    return str(scenarios)


def save_config(config, args):
    """Save configuration and args to a folder with timestamp."""
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    variant_tag = build_variant_tag(args)
    scenario_tag = build_scenario_tag(config, args)

    foldername = f"./save/simulation_{variant_tag}_{scenario_tag}_{current_time}/"
    print("model folder:", foldername)
    os.makedirs(foldername, exist_ok=True)

    args_dict = vars(args)

    merged_config = config.copy()
    merged_config["args"] = args_dict

    with open(foldername + "config.json", "w") as f:
        json.dump(merged_config, f, indent=4)

    return foldername


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSDI")

    parser.add_argument("--config", type=str, default="base_scenmap.yaml")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--testmissingratio", type=float, default=0.1)
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--data_length", type=int, default=100)

    parser.add_argument(
        "--model_variant",
        type=str,
        default="baseline",
        choices=VARIANT_CHOICES,
    )

    parser.add_argument(
        "--ablation_order",
        type=str,
        default="reordered",
        choices=["legacy", "reordered"],
        help="legacy = goal guidance first; reordered = goal guidance last",
    )

    parser.add_argument("--max_neighbors", type=int, default=8)
    parser.add_argument("--eval_collision", action="store_true")

    parser.add_argument("--force_random_strategy", action="store_true")
    parser.add_argument("--force_random_target_know_first", action="store_true")
    parser.add_argument("--scenarios", nargs="+", default=None)

    args = parser.parse_args()
    print(args)

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config.setdefault("model", {})
    config.setdefault("dataset", {})

    config["model"]["test_missing_ratio"] = args.testmissingratio

    if args.scenarios is not None:
        config["dataset"]["scenarios"] = args.scenarios

    model_family = configure_variant(config, args.model_variant, args.ablation_order)

    if args.force_random_strategy and args.force_random_target_know_first:
        raise ValueError(
            "Do not use --force_random_strategy and "
            "--force_random_target_know_first at the same time."
        )

    if args.force_random_strategy:
        config["model"]["target_strategy"] = "random"
        config["dataset"]["missing_strategy"] = "random"
        config["dataset"]["missing_ratio"] = 0.5

    if args.force_random_target_know_first:
        config["model"]["target_strategy"] = "random"
        config["dataset"]["missing_strategy"] = "end"
        config["dataset"]["missing_ratio"] = 0.5

    print(json.dumps(config, indent=4))

    foldername = args.modelfolder or save_config(config, args)

    if model_family == "base":
        train_loader, valid_loader, test_loader = get_dataloader(
            scenarios=config["dataset"]["scenarios"],
            data_length=args.data_length,
            seed=args.seed,
            batch_size=config["train"]["batch_size"],
            zero_based_position=True,
            load_scenario_map=True,
            missing_strategy=config["dataset"].get("missing_strategy", "end"),
            missing_ratio=config["dataset"].get("missing_ratio", 0.5),
        )

        model = CSDI_SimulationScenmap(config, args.device).to(args.device)

    elif model_family == "social":
        train_loader, valid_loader, test_loader = get_social_dataloader(
            scenarios=config["dataset"]["scenarios"],
            data_length=args.data_length,
            seed=args.seed,
            batch_size=config["train"]["batch_size"],
            zero_based_position=True,
            load_scenario_map=True,
            max_neighbors=args.max_neighbors,
            gen_sdf=config["model"].get("add_collision_loss", False),
            missing_strategy=config["dataset"].get("missing_strategy", "end"),
            missing_ratio=config["dataset"].get("missing_ratio", 0.5),
        )

        model = CSDI_SocialFusionScenmap(config, args.device).to(args.device)

    else:
        raise ValueError(f"Unknown model family: {model_family}")

    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

    preproc_tag = "nonaug"

    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=1,
        foldername=foldername,
        file_tag=f"{preproc_tag}_{args.model_variant}",
        eval_collision=args.eval_collision,
    )