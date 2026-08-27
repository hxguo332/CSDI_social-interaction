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


def select_validity_aware_sample(
    samples,
    eval_points,
    scen_maps_batch=None,
    scenmap_scale=None,
    mode: str = "normalized",
    neighbor_data=None,
    neighbor_mask=None,
    social_margin: float = 0.04,
):
    """
    Select one trajectory per case from stochastic diffusion samples.

    Priority:
      1. lowest invalid/collision step count;
      2. among ties, closest to the element-wise sample median.

    samples: [B, S, L, 2]
    eval_points: [B, L, 2]
    scen_maps_batch: optional [B, 3, H, W]
    """
    median = samples.median(dim=1).values
    if scen_maps_batch is None:
        return median

    B, S, L, _ = samples.shape
    collision_mask = (scen_maps_batch[:, 0] > 0) | (scen_maps_batch[:, 1] > 0)
    H = scen_maps_batch.shape[2]
    W = scen_maps_batch.shape[3]

    if mode == "normalized":
        x_float = samples[..., 0] * W
        y_float = samples[..., 1] * H
    elif mode == "unnormalized":
        if not isinstance(scenmap_scale, torch.Tensor):
            scenmap_scale = torch.full((B, 2), float(scenmap_scale), device=samples.device, dtype=samples.dtype)
        else:
            if scenmap_scale.dim() == 1:
                scenmap_scale = scenmap_scale.view(B, 1).repeat(1, 2)
            elif scenmap_scale.shape[-1] == 1:
                scenmap_scale = scenmap_scale.view(B, 1).repeat(1, 2)
        eps = torch.finfo(samples.dtype).eps
        x_float = samples[..., 0] / (scenmap_scale[:, 0].view(B, 1, 1) + eps)
        y_float = samples[..., 1] / (scenmap_scale[:, 1].view(B, 1, 1) + eps)
    else:
        raise ValueError(f"Unsupported mode for select_validity_aware_sample: {mode}")

    valid_t = eval_points[..., 0] > 0
    valid_t_s = valid_t.unsqueeze(1).expand(-1, S, -1)

    oob = (x_float < 0) | (x_float >= (W - 1e-6)) | (y_float < 0) | (y_float >= (H - 1e-6))
    x = x_float.long().clamp(0, W - 1)
    y = y_float.long().clamp(0, H - 1)
    bidx = torch.arange(B, device=samples.device).view(B, 1, 1).expand(B, S, L)
    map_collision = collision_mask[bidx, y, x] | oob
    invalid = map_collision & valid_t_s

    if neighbor_data is not None and neighbor_mask is not None:
        if not isinstance(neighbor_data, torch.Tensor):
            neighbor_data = torch.from_numpy(neighbor_data)
        if not isinstance(neighbor_mask, torch.Tensor):
            neighbor_mask = torch.from_numpy(neighbor_mask)
        neighbor_data = neighbor_data.to(samples.device).float()
        neighbor_mask = neighbor_mask.to(samples.device).float()
        social_dist = torch.linalg.norm(samples.unsqueeze(2) - neighbor_data.unsqueeze(1), dim=-1)
        social_collision = (social_dist < social_margin) & (neighbor_mask.unsqueeze(1) > 0)
        social_collision = social_collision.any(dim=2) & valid_t_s
        invalid = invalid | social_collision

    invalid_count = invalid.float().sum(dim=-1)  # [B,S]
    median_dist = torch.linalg.norm((samples - median.unsqueeze(1)) * eval_points.unsqueeze(1), dim=-1).sum(dim=-1)
    # Large constant makes invalid count primary and median distance secondary.
    score = invalid_count * 1e6 + median_dist
    best_idx = score.argmin(dim=1)
    return samples[torch.arange(B, device=samples.device), best_idx]


def save_generated_outputs(
    foldername,
    nsample,
    all_target,
    all_evalpoint,
    all_observed_point,
    all_observed_time,
    all_generated_samples,
    scaler,
    mean_scaler,
    mode=None,
    tag=None,
    behavior_context=None,
):
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
                behavior_context,
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
        self.total_collisions = {"obstacle": 0, "social": 0, "any": 0}
        self.total_paths = 0
        self.invalid_steps = {"obstacle": 0, "social": 0, "any": 0}
        self.total_steps = 0

        self.scenmap_scale = scenmap_scale

    def update(
        self,
        samples_batch,
        scen_maps_batch,
        eval_points,
        scenmap_scale,
        mode: str = "normalized",
        neighbor_data=None,
        neighbor_mask=None,
        social_margin: float = 0.04,
    ):
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

        # Only evaluate predicted / imputed target points. Do not multiply
        # non-evaluation coordinates by zero before the out-of-bounds check,
        # otherwise masked-out points may be incorrectly counted as collisions.
        valid_eval = eval_points[..., 0] > 0

        # Out-of-bounds: before clamping, flag as collision only at eval points.
        oob_x = ((x_float < 0) | (x_float >= (W - 1e-6))) & valid_eval
        oob_y = ((y_float < 0) | (y_float >= (H - 1e-6))) & valid_eval
        collision_position = oob_x | oob_y

        # Discretize to integer pixel indices
        x = x_float.long().clamp(0, W - 1)
        y = y_float.long().clamp(0, H - 1)

        # Generate batch indices
        batch_indices = torch.arange(batch_size, device=samples_batch.device).view(-1, 1).expand(-1, sample_length)

        # Collisions against obstacle mask under eval points only.
        obstacle_collision = (collision_mask[batch_indices, y, x] & valid_eval) | collision_position
        social_collision = torch.zeros_like(obstacle_collision)

        if neighbor_data is not None and neighbor_mask is not None:
            if not isinstance(neighbor_data, torch.Tensor):
                neighbor_data = torch.from_numpy(neighbor_data)
            if not isinstance(neighbor_mask, torch.Tensor):
                neighbor_mask = torch.from_numpy(neighbor_mask)
            neighbor_data = neighbor_data.to(samples_batch.device).float()
            neighbor_mask = neighbor_mask.to(samples_batch.device).float()
            social_dist = torch.linalg.norm(samples_batch.unsqueeze(1) - neighbor_data, dim=-1)
            social_collision = ((social_dist < social_margin) & (neighbor_mask > 0)).any(dim=1) & valid_eval

        collision_masks = {
            "obstacle": obstacle_collision,
            "social": social_collision,
            "any": obstacle_collision | social_collision,
        }
        for name, collision in collision_masks.items():
            self.total_collisions[name] += torch.any(collision, dim=1).sum().item()
            self.invalid_steps[name] += collision.sum().item()
        self.total_paths += batch_size
        self.total_steps += valid_eval.sum().item()

    def compute_metrics(self):
        """
        Compute the average collision rate and invalid rate across all batches.

        Returns:
            collision_rate (float): Average collision rate across all batches.
            invalid_rate (float): Average invalid rate across all batches.
        """
        # Calculate the average collision rate across all batches
        return {
            f"{name}_{metric}": value / denominator if denominator else 0.0
            for name in ("obstacle", "social", "any")
            for metric, value, denominator in (
                ("collision_rate", self.total_collisions[name], self.total_paths),
                ("invalid_rate", self.invalid_steps[name], self.total_steps),
            )
        }


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

        collision_evaluators = {
            name: CollisionEvaluator() for name in ("best_of_30", "median", "expected")
        } if eval_collision else None

        all_target, all_observed_point, all_observed_time = [], [], []
        all_evalpoint, all_generated_samples = [], []
        all_neighbor_data, all_neighbor_mask, all_conflict_features = [], [], []
        all_scen_map, all_goal_heatmap = [], []

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

                selection_scen_map = None
                selection_scen_scale = None
                if isinstance(test_batch, dict):
                    batch_map = test_batch.get("scen_map_raw", test_batch.get("scen_map", None))
                    if batch_map is not None:
                        bm = batch_map if isinstance(batch_map, torch.Tensor) else torch.from_numpy(batch_map)
                        if bm.dim() == 4 and bm.shape[-1] == 3:
                            bm = bm.permute(0, 3, 1, 2)
                        selection_scen_map = bm.to(samples.device).float()

                    if "scen_map_scale" in test_batch:
                        sms = test_batch["scen_map_scale"]
                        selection_scen_scale = sms.to(samples.device).float() if isinstance(sms, torch.Tensor) else torch.from_numpy(sms).to(samples.device).float()

                ce_mode = mode if mode is not None else "normalized"
                neighbor_data_for_selection = test_batch.get("neighbor_data", None) if isinstance(test_batch, dict) else None
                neighbor_mask_for_selection = test_batch.get("neighbor_mask", None) if isinstance(test_batch, dict) else None
                samples_median = select_validity_aware_sample(
                    samples,
                    eval_points,
                    scen_maps_batch=selection_scen_map,
                    scenmap_scale=selection_scen_scale,
                    mode=ce_mode,
                    neighbor_data=neighbor_data_for_selection,
                    neighbor_mask=neighbor_mask_for_selection,
                    social_margin=getattr(model, "social_margin", 0.04),
                )

                if collision_evaluators is not None:
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
                        neighbor_data = test_batch.get("neighbor_data", None) if isinstance(test_batch, dict) else None
                        neighbor_mask = test_batch.get("neighbor_mask", None) if isinstance(test_batch, dict) else None
                        social_margin = getattr(model, "social_margin", 0.04)
                        median_trajectory = samples.median(dim=1).values
                        _, S, _, _ = samples.shape
                        trajectories = {
                            "best_of_30": samples_median,
                            "median": median_trajectory,
                        }
                        for metric_name, trajectory in trajectories.items():
                            collision_evaluators[metric_name].update(
                                trajectory,
                                scen_map_tensor,
                                eval_points,
                                scen_maps_scale_tensor,
                                mode=ce_mode,
                                neighbor_data=neighbor_data,
                                neighbor_mask=neighbor_mask,
                                social_margin=social_margin,
                            )
                        for sample_idx in range(S):
                            collision_evaluators["expected"].update(
                                samples[:, sample_idx],
                                scen_map_tensor,
                                eval_points,
                                scen_maps_scale_tensor,
                                mode=ce_mode,
                                neighbor_data=neighbor_data,
                                neighbor_mask=neighbor_mask,
                                social_margin=social_margin,
                            )

                # Append results to the respective lists
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                if isinstance(test_batch, dict):
                    if "neighbor_data" in test_batch:
                        all_neighbor_data.append(test_batch["neighbor_data"].detach().cpu())
                    if "neighbor_mask" in test_batch:
                        all_neighbor_mask.append(test_batch["neighbor_mask"].detach().cpu())
                    if "conflict_features" in test_batch:
                        all_conflict_features.append(test_batch["conflict_features"].detach().cpu())
                    # Do not store scen_map / goal_heatmap per sample; they are large and can exceed quota.
                    # Visualization can reload maps separately if needed.

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
        behavior_context = {
            "neighbor_data": torch.cat(all_neighbor_data, dim=0) if all_neighbor_data else None,
            "neighbor_mask": torch.cat(all_neighbor_mask, dim=0) if all_neighbor_mask else None,
            "conflict_features": torch.cat(all_conflict_features, dim=0) if all_conflict_features else None,
            "scen_map": None,
            "goal_heatmap": None,
        }

        save_generated_outputs(
            foldername,
            nsample,
            torch.cat(all_target, dim=0),
            torch.cat(all_evalpoint, dim=0),
            torch.cat(all_observed_point, dim=0),
            torch.cat(all_observed_time, dim=0),
            torch.cat(all_generated_samples, dim=0),
            scaler,
            mean_scaler,
            mode,
            tag=file_tag,
            behavior_context=behavior_context,
        )

        # Compute CRPS metrics
        CRPS, CRPS_sum = compute_crps_metrics(torch.cat(all_target, dim=0), torch.cat(all_generated_samples, dim=0),
                                              torch.cat(all_evalpoint, dim=0), mean_scaler, scaler)
        # Compute collision metrics if applicable


        # Save evaluation metrics
        RMSE = np.sqrt(mse_total / evalpoints_total)
        MAE = mae_total / evalpoints_total
        if collision_evaluators is not None:
            collision_metrics = {
                f"{selection}_{metric}": value
                for selection, evaluator in collision_evaluators.items()
                for metric, value in evaluator.compute_metrics().items()
            }
            for key, value in collision_metrics.items():
                print(f"{key}: {value}")
            metrics_dict = {
                "RMSE": RMSE,
                "MAE": MAE,
                "CRPS": CRPS,
                "CRPS_sum": CRPS_sum,
                # Backward-compatible fields use the original best-of-30 any-collision definition.
                "Collision Rate": collision_metrics["best_of_30_any_collision_rate"],
                "Invalid Rate": collision_metrics["best_of_30_any_invalid_rate"],
                **collision_metrics,
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
