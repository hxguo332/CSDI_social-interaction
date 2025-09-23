import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
from typing import Optional


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
    batch_sampler_train=None,
    batch_sampler_valid=None,
):
    # Set up optimizer with param groups: smaller LR for ResNet backbone to stabilize fine-tuning
    base_lr = float(config["lr"])
    backbone_lr_mult = float(config.get("encoder_backbone_lr_mult", 0.1))

    enc_backbone_params = []
    enc_head_params = []
    if hasattr(model, "emb_scenmap"):
        enc = model.emb_scenmap
        if hasattr(enc, "backbone"):
            enc_backbone_params = [p for p in enc.backbone.parameters() if p.requires_grad]
        if hasattr(enc, "fc"):
            enc_head_params = [p for p in enc.fc.parameters() if p.requires_grad]

    enc_backbone_ids = set(id(p) for p in enc_backbone_params)
    enc_head_ids = set(id(p) for p in enc_head_params)
    excluded = enc_backbone_ids | enc_head_ids
    other_params = [p for p in model.parameters() if id(p) not in excluded]

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr})
    if enc_head_params:
        param_groups.append({"params": enc_head_params, "lr": base_lr})
    if enc_backbone_params:
        param_groups.append({"params": enc_backbone_params, "lr": max(base_lr * backbone_lr_mult, 1e-8)})

    optimizer = Adam(param_groups if param_groups else model.parameters(), lr=base_lr, weight_decay=1e-6)

    # One-time log: optimizer param groups
    try:
        print("[optimizer] param groups:")
        for i, g in enumerate(optimizer.param_groups):
            lr_g = g.get("lr", base_lr)
            print(f"  group[{i}]: lr={lr_g:.3e}, params={len(g['params'])}")
    except Exception:
        pass
    if foldername != "":
        output_path = foldername + "/model.pth"
    else:
        output_path = None

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    debug_grad_logged = False
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        if batch_sampler_train is not None:
            batch_sampler_train.set_epoch(epoch_no)
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(train_batch)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Skipping batch {batch_no} due to NaN/Inf loss.")
                    continue

                loss.backward()

                # One-time gradient norm logging to confirm training is active
                if not debug_grad_logged:
                    try:
                        grad_out2 = None
                        if hasattr(model, "diffmodel") and hasattr(model.diffmodel, "output_projection2"):
                            p = model.diffmodel.output_projection2.weight
                            if p.grad is not None:
                                grad_out2 = p.grad.detach().norm().item()

                        grad_enc_fc = None
                        if hasattr(model, "emb_scenmap") and hasattr(model.emb_scenmap, "fc"):
                            p = model.emb_scenmap.fc.weight
                            if p.grad is not None:
                                grad_enc_fc = p.grad.detach().norm().item()

                        grad_enc_backbone = None
                        if hasattr(model, "emb_scenmap") and hasattr(model.emb_scenmap, "backbone"):
                            for p in model.emb_scenmap.backbone.parameters():
                                if p.requires_grad and p.grad is not None:
                                    grad_enc_backbone = p.grad.detach().norm().item()
                                    break

                        print("[grad] norms: output_projection2=", grad_out2, 
                              ", enc_fc=", grad_enc_fc, ", enc_backbone=", grad_enc_backbone)
                    except Exception:
                        pass
                    debug_grad_logged = True
                avg_loss += loss.item()
                optimizer.step()

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 1000000
            loss_sum = 0
            if batch_sampler_valid is not None:
                batch_sampler_valid.set_epoch(epoch_no)
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        loss_sum += loss.item()
                        avg_loss_valid = loss_sum / batch_no
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid,
                    "at",
                    epoch_no,
                )

    if output_path is not None:
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    ).item()


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS / len(quantiles)

def process_batch_data(output):
    """
    Process and permute batch data for evaluation.

    Args:
        output: Output from the model's evaluation method.

    Returns:
        Processed tensors for samples, target, eval_points, observed_points, and observed_time.
    """
    samples, c_target, eval_points, observed_points, observed_time = output

    # Permute dimensions for further processing
    samples = samples.permute(0, 1, 3, 2)  # (B, nsample, L, K)
    c_target = c_target.permute(0, 2, 1)  # (B, L, K)
    eval_points = eval_points.permute(0, 2, 1)
    observed_points = observed_points.permute(0, 2, 1)

    return samples, c_target, eval_points, observed_points, observed_time, 


def compute_batch_metrics(samples_median, c_target, eval_points, scaler):
    """
    Compute MSE and MAE for a batch.

    Args:
        samples_median: Median of generated samples.
        c_target: Ground truth target values.
        eval_points: Evaluation points mask.
        scaler: Scaling factor for the target values.

    Returns:
        mse_current: Mean Squared Error for the batch.
        mae_current: Mean Absolute Error for the batch.
    """
    mse_current = (
        ((samples_median - c_target) * eval_points) ** 2
    ) * (scaler ** 2)
    mae_current = (
        torch.abs((samples_median - c_target) * eval_points)
    ) * scaler

    return mse_current.sum().item(), mae_current.sum().item()


def save_generated_outputs(foldername, nsample, all_target, all_evalpoint, all_observed_point, all_observed_time, all_generated_samples, scaler, mean_scaler, mode=None, tag=None):
    """
    Save generated outputs to a pickle file.

    Args:
        foldername: Directory to save the file.
        nsample: Number of samples generated.
        all_target, all_evalpoint, all_observed_point, all_observed_time, all_generated_samples: Evaluation data.
        scaler: Scaling factor for the target values.
        mean_scaler: Mean scaler for the target values.
    """
    suffix_parts = []
    if mode is not None:
        suffix_parts.append(str(mode))
    if tag is not None and str(tag):
        suffix_parts.append(str(tag))
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    file_path = foldername + f"/generated_outputs_nsample{nsample}{suffix}.pk"
    with open(file_path, "wb") as f:
        pickle.dump(
            [
                all_generated_samples,
                all_target,
                all_evalpoint,
                all_observed_point,
                all_observed_time,
                scaler,
                mean_scaler,
            ],
            f,
        )


def compute_crps_metrics(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler):
    """
    Compute CRPS and CRPS_sum metrics.

    Args:
        all_target: Ground truth target values.
        all_generated_samples: Generated samples from the model.
        all_evalpoint: Evaluation points mask.
        mean_scaler: Mean scaler for the target values.
        scaler: Scaling factor for the target values.

    Returns:
        CRPS: Continuous Ranked Probability Score.
        CRPS_sum: Summed CRPS metric.
    """
    CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)
    CRPS_sum = calc_quantile_CRPS_sum(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)
    return CRPS, CRPS_sum

class CollisionEvaluator:
    """
    A class to compute and store collision metrics (collision rate and invalid rate)
    across multiple batches, and calculate the average metrics at the end.

    Attributes:
        total_collisions (int): Total number of collisions across all batches.
        total_paths (int): Total number of paths across all batches.
        invalid_rate_all (list): List of invalid rates for each batch.
    """

    def __init__(self, scenmap_scale=10):
        self.total_collisions = 0
        self.total_paths = 0

        self.invalid_rate_all = []

        self.scenmap_scale = scenmap_scale

    def update(self, samples_batch, scen_maps_batch, eval_points, scenmap_scale, mode: str = "normalized"):
        """
        Update the collision metrics with a new batch of data.

        Args:
            samples_batch (torch.Tensor): Predicted paths. Shape: (batch_size, sample_length, 2).
            scen_maps_batch (torch.Tensor): Scenario map with obstacles and forbidden areas.
                - Channel 1 (Red): Obstacles (rectangles, circles) & optionally standing persons.
                - Channel 2 (Green): Walls & forbidden areas.
                - Channel 3 (Blue): Entrances.
            eval_points (torch.Tensor): Evaluation points mask. Shape: (batch_size, sample_length, 2).
            scenmap_scale (int, float, or torch.Tensor): Per-axis world-per-pixel (wpp) used when converting
                between world/track units and map pixels. For AugmentedSimulationDataset this is
                typically a (B, 2) tensor; often x/y are equal. Only used when
                mode == "unnormalized". Ignored for "normalized" mode.
            mode (str): "normalized" if samples are in [0,1] map coordinates;
                "unnormalized" if samples are in world/track units scaled from the map.
        """
        # Build collision mask from scenario map channels (B, C, H, W)
        collision_mask = (scen_maps_batch[:, 0] > 0) | (scen_maps_batch[:, 1] > 0)

        batch_size = samples_batch.shape[0]
        sample_length = samples_batch.shape[1]

        H = scen_maps_batch.shape[2]
        W = scen_maps_batch.shape[3]

        # Convert coordinates to pixel indices depending on mode
        if mode == "normalized":
            # samples in [0,1] relative to map width/height; dataset already in image coords
            x_float = samples_batch[..., 0] * W
            y_float = samples_batch[..., 1] * H
        elif mode == "unnormalized":
            # samples are in world/track units after being scaled by scenmap_scale during generation
            # Convert back to pixel units by dividing by scenmap_scale per axis
            if not isinstance(scenmap_scale, torch.Tensor):
                # broadcast scalar to (B, 2)
                scenmap_scale = torch.full((batch_size, 2), float(scenmap_scale), device=samples_batch.device, dtype=samples_batch.dtype)
            else:
                # Expect shape (B, 2) or (B,)
                if scenmap_scale.dim() == 1:
                    scenmap_scale = scenmap_scale.view(batch_size, 1).repeat(1, 2)
                elif scenmap_scale.shape[-1] == 1:
                    scenmap_scale = scenmap_scale.view(batch_size, 1).repeat(1, 2)
            scale_x = scenmap_scale[:, 0].view(batch_size, 1)
            scale_y = scenmap_scale[:, 1].view(batch_size, 1)
            # Avoid divide-by-zero
            eps = torch.finfo(samples_batch.dtype).eps
            x_float = samples_batch[..., 0] / (scale_x + eps)
            y_float = samples_batch[..., 1] / (scale_y + eps)
        else:
            raise ValueError(f"Unsupported mode for CollisionEvaluator.update: {mode}")

        # Apply eval_points mask to avoid unintended indices; keep float for OOB checks
        x_float = x_float * eval_points[..., 0]
        y_float = y_float * eval_points[..., 1]

        # Out-of-bounds: before clamping, flag as collision if outside map
        oob_x = (x_float < 0) | (x_float >= (W - 1e-6))
        oob_y = (y_float < 0) | (y_float >= (H - 1e-6))
        collision_position = oob_x | oob_y

        # Discretize to integer pixel indices
        x = x_float.long().clamp(0, W - 1)
        y = y_float.long().clamp(0, H - 1)

        # Generate batch indices
        batch_indices = torch.arange(batch_size, device=samples_batch.device).view(-1, 1).expand(-1, sample_length)

        # Collisions against obstacle mask under eval points
        collision = collision_mask[batch_indices, y, x] * eval_points[..., 0]
        collision = collision.bool() | collision_position

        # Calculate collision rate for the batch, calculated by each sample
        batch_collisions = torch.any(collision, dim=1).sum().item()  # Number of paths with at least one collision

        # Calculate invalid rate for the batch
        batch_invalid_steps = collision.sum(dim=-1)  # Total number of invalid steps
        batch_total_steps = eval_points[...,0].sum(dim=-1)  # Total number of steps under eval_points

        batch_invalid_rate = batch_invalid_steps / batch_total_steps  # Invalid rate for each path
        # Handle NaN values
        batch_invalid_rate[batch_total_steps == 0] = 0.0  # Set invalid rate to 0 if total steps is 0
        batch_invalid_rate = batch_invalid_rate.tolist()  # Convert to list for easier handling

        # Update the totals
        self.invalid_rate_all.extend(batch_invalid_rate)
        self.total_collisions += batch_collisions
        self.total_paths += batch_size

    def compute_metrics(self):
        """
        Compute the average collision rate and invalid rate across all batches.

        Returns:
            collision_rate (float): Average collision rate across all batches.
            invalid_rate (float): Average invalid rate across all batches.
        """
        # Calculate the average collision rate across all batches
        collision_rate = self.total_collisions / self.total_paths if self.total_paths > 0 else 0.0

        # Calculate the average invalid rate across all batches
        invalid_rate = np.mean(self.invalid_rate_all) if len(self.invalid_rate_all) > 0 else 0.0
        return collision_rate, invalid_rate


def save_evaluation_metrics(foldername, nsample, metrics_dict, mode=None, tag=None):
    """
    Save evaluation metrics to a CSV file.

    Args:
        foldername: Directory to save the file.
        nsample: Number of samples generated.
        metrics_dict: Dictionary containing evaluation metrics (RMSE, MAE, CRPS, etc.).
        mode: Sampling mode, either "normalized" or "unnormalized" (optional).
    """
    suffix_parts = []
    if mode is not None:
        suffix_parts.append(str(mode))
    if tag is not None and str(tag):
        suffix_parts.append(str(tag))
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    result_path = foldername + f"/result_nsample{nsample}{suffix}.csv"
    with open(result_path, "w") as f:
        f.write("Metric,Value\n")
        for key, value in metrics_dict.items():
            f.write(f"{key},{value}\n")


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", eval_collision=False, mode=None, file_tag=None):
    """
    Evaluate the model on the test dataset and compute various metrics.

    Args:
        model: The trained model to evaluate.
        test_loader: DataLoader for the test dataset.
        nsample: Number of samples to generate for evaluation.
        scaler: Scaling factor for the target values.
        mean_scaler: Mean scaler for the target values.
        foldername: Directory to save evaluation results and generated outputs.
        eval_collision: Whether to compute collision metrics (collision rate and invalid rate).
        mode: Sampling mode, either "normalized" or "unnormalized". If None, do not pass mode to model.evaluate.

    Returns:
        None. Saves evaluation metrics and generated outputs to files.
    """
    with torch.no_grad():
        model.eval()
        mse_total, mae_total, evalpoints_total = 0, 0, 0

        collision_evaluator = None
        if eval_collision:
            collision_evaluator = CollisionEvaluator()

        all_target, all_observed_point, all_observed_time = [], [], []
        all_evalpoint, all_generated_samples = [], []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                if batch_no == 1:
                    print("Evaluating with nsample =", nsample)
                    print("  mode =", mode)
                    print("  eval_collision =", eval_collision)
                    if eval_collision:
                        if "scen_map_raw" in test_batch:
                            print("  Using scen_map_raw from dataloader for collision evaluation")
                        elif "scen_map" in test_batch:
                            print("  Using scen_map from dataloader for collision evaluation")
                    else:
                        print("  Collision evaluation disabled")

                # Best-effort call to model.evaluate compatible with both original and scenemap models
                try:
                    if mode is not None:
                        output = model.evaluate(test_batch, nsample, mode=mode)
                    else:
                        output = model.evaluate(test_batch, nsample)
                except TypeError:
                    # Fallback for models without extra kwargs
                    output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = process_batch_data(output)
                # Compute the median of the generated samples
                samples_median = samples.median(dim=1).values

                if collision_evaluator is not None:
                    # Always use scen_map from batch (raw or original). Do not use scen_map_out returned by the model,
                    # as it is normalized and unsuitable for collision evaluation.
                    scen_map_tensor = None
                    scen_maps_scale_tensor = None

                    batch_map = None
                    if isinstance(test_batch, dict) and ("scen_map_raw" in test_batch):
                        batch_map = test_batch["scen_map_raw"]
                    elif isinstance(test_batch, dict) and ("scen_map" in test_batch):
                        batch_map = test_batch["scen_map"]

                    if batch_map is not None:
                        if isinstance(batch_map, torch.Tensor):
                            bm = batch_map
                        else:
                            bm = torch.from_numpy(batch_map)
                        # Expect (B,H,W,3); convert to (B,3,H,W)
                        if bm.dim() == 4 and bm.shape[-1] == 3:
                            bm = bm.permute(0, 3, 1, 2)
                        scen_map_tensor = bm.to(samples_median.device).float()

                    if isinstance(test_batch, dict) and ("scen_map_scale" in test_batch):
                        sms = test_batch["scen_map_scale"]
                        if isinstance(sms, torch.Tensor):
                            scen_maps_scale_tensor = sms.to(samples_median.device).float()
                        else:
                            scen_maps_scale_tensor = torch.from_numpy(sms).to(samples_median.device).float()

                    if scen_maps_scale_tensor is None and eval_collision:
                        print("⚠️ Warning: scen_map_scale not found in test batch; The evaluation is stopped.")

                    ce_mode = mode if mode is not None else "normalized"
                    if (scen_map_tensor is not None) and (scen_maps_scale_tensor is not None):
                        collision_evaluator.update(samples_median, scen_map_tensor, eval_points, scen_maps_scale_tensor, mode=ce_mode)

                # Append results to the respective lists
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                # Compute MSE and MAE for the current batch
                mse_current, mae_current = compute_batch_metrics(samples_median, c_target, eval_points, scaler)

                # Accumulate MSE, MAE, and evaluation points
                mse_total += mse_current
                mae_total += mae_current
                evalpoints_total += eval_points.sum().item()

                # Update progress bar
                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

        # Save generated outputs
        save_generated_outputs(foldername, nsample, torch.cat(all_target, dim=0), torch.cat(all_evalpoint, dim=0),
                               torch.cat(all_observed_point, dim=0), torch.cat(all_observed_time, dim=0),
                               torch.cat(all_generated_samples, dim=0), scaler, mean_scaler, mode, tag=file_tag)

        # Compute CRPS metrics
        CRPS, CRPS_sum = compute_crps_metrics(torch.cat(all_target, dim=0), torch.cat(all_generated_samples, dim=0),
                                              torch.cat(all_evalpoint, dim=0), mean_scaler, scaler)
        # Compute collision metrics if applicable


        # Save evaluation metrics
        RMSE = np.sqrt(mse_total / evalpoints_total)
        MAE = mae_total / evalpoints_total
        if collision_evaluator is not None:
            collision_rate, invalid_rate = collision_evaluator.compute_metrics()
            print("Collision Rate:", collision_rate)
            print("Invalid Rate:", invalid_rate)
            metrics_dict = {
                "RMSE": RMSE,
                "MAE": MAE,
                "CRPS": CRPS,
                "CRPS_sum": CRPS_sum,
                "Collision Rate": collision_rate,
                "Invalid Rate": invalid_rate,
            }
        else:
            metrics_dict = {
                "RMSE": RMSE,
                "MAE": MAE,
                "CRPS": CRPS,
                "CRPS_sum": CRPS_sum,
            }
        save_evaluation_metrics(foldername, nsample, metrics_dict, mode, tag=file_tag)

        # Print evaluation metrics
        print("RMSE:", RMSE)
        print("MAE:", MAE)
        print("CRPS:", CRPS)
        print("CRPS_sum:", CRPS_sum)


def build_preproc_tag_from_split_cfg(split_cfg: dict, fallback_name: Optional[str] = None) -> Optional[str]:
    """
    Build a concise preprocessing tag from a split's dataloader config.

    Examples:
    - augmented + pad_and_resize + [[512,512]] -> "pad_and_resize_512x512"
    - augmented + crop_and_resize + [[384,384],[512,512]] -> "crop_and_resize_multi2"
    - nonaugmented/nonaugmented_scenario_batch/scenario_batch -> "nonaug"

    Returns a short string or None if it can't be inferred.
    """
    if not isinstance(split_cfg, dict):
        return fallback_name

    dl_type = split_cfg.get("dataloader")
    if dl_type in ("nonaugmented", "nonaugmented_scenario_batch", "scenario_batch"):
        return "nonaug"

    if dl_type == "augmented":
        mode = split_cfg.get("augmentation_mode", "pad_and_resize")
        sizes = split_cfg.get("resize_options", None)
        if isinstance(sizes, (list, tuple)) and len(sizes) > 0:
            # Normalize sizes list to list of (h,w)
            norm = []
            for s in sizes:
                try:
                    h, w = int(s[0]), int(s[1])
                    norm.append(f"{h}x{w}")
                except Exception:
                    continue
            if len(norm) == 1:
                return f"{mode}_{norm[0]}"
            elif len(norm) > 1:
                return f"{mode}_multi{len(norm)}"
        return mode

    return fallback_name
