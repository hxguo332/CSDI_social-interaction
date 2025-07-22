import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_SimulationScenmap
from dataset_augmented_simulation import get_dataloader
from utils import train, evaluate

def save_config(config, args):
    """Save configuration and args to a folder with timestamp."""
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = "./save/simulation" + "_" + current_time + "/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)

    # Convert args to dictionary
    args_dict = vars(args)

    # Merge args_dict into config under a new key called "args"
    merged_config = config.copy()
    merged_config["args"] = args_dict  # Store all args under the key "args"

    # Save the merged dictionary to config.json
    with open(foldername + "config.json", "w") as f:
        json.dump(merged_config, f, indent=4)
    
    return foldername  # Return foldername to be used later

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSDI")
    parser.add_argument("--config", type=str, default="base_scenmap.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device for Attack')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--testmissingratio", type=float, default=0.1)
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--data_length", type=int, default=100)

    args = parser.parse_args()
    print(args)

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["test_missing_ratio"] = args.testmissingratio

    print(json.dumps(config, indent=4))

    # Call the function to save the config and get the folder name
    foldername = save_config(config, args)

    train_loader, valid_loader, test_loader = get_dataloader(
        scenarios=config["dataset"]["scenarios"],
        data_length=args.data_length,
        seed=args.seed,
        batch_size=config["train"]["batch_size"],
        zero_based_position=True,
        load_scenario_map=True,
        #missing_ratio=config["model"]["test_missing_ratio"],
    )

    model = CSDI_SimulationScenmap(config, args.device).to(args.device)

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

    evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
