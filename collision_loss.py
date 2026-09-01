import torch
import torch.nn.functional as F
from typing import Callable, Dict

def compute_collision_loss(
    xy: torch.Tensor,
    ta_time_mask: torch.Tensor,
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    w_obs: float = 1.0,
    w_clear: float = 0.0,
    margin: float = 0.0,
    reduction: str = "mean",
    detach_stats: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Compute the collision (geometry) loss and related statistics for trajectory samples.

    Parameters
    ----------
    xy : torch.Tensor, shape [B, L, 2]
        Predicted 2D coordinates (typically denoised estimates x0_hat for each time step).
        B = batch size, L = sequence length, 2 = (x, y).
    ta_time_mask : torch.Tensor, shape [B, L]
        Binary mask (1 = missing/imputed time step, 0 = observed).
        The loss and statistics are computed only where mask == 1.
    sdf_fn : Callable
        A signed distance function: sdf_fn(xy) -> d, where d has shape [B, L].
        Positive values mean inside the free space, negative values mean inside obstacles.
    w_obs : float, default=1.0
        Weight for the obstacle penetration term ReLU(-d)^2.
    w_clear : float, default=0.0
        Weight for the clearance (safe margin) term ReLU(margin - d)^2.
    margin : float, default=0.0
        Desired safety distance from obstacles. Units must match the scale of sdf_fn.
    reduction : {"mean", "sum", "none"}, default="mean"
        Reduction method for the loss aggregation:
        - "mean": average over masked valid points
        - "sum": sum over masked valid points
        - "none": return pointwise loss tensors
    detach_stats : bool, default=True
        If True, the returned statistics (collision_rate, mean_clearance) are detached
        from the computational graph and will not backpropagate gradients.

    Returns
    -------
    dict
        A dictionary with the following keys:
          - "loss": total geometric loss (scalar or tensor depending on reduction)
          - "L_obs": obstacle penetration component
          - "L_clear": clearance (margin) component
          - "L_path": worst-point path collision component
          - "collision_rate": ratio of points with d < 0 within masked steps
          - "mean_clearance": mean positive distance to obstacles within masked steps
    """
    assert xy.dim() == 3 and xy.size(-1) == 2, "xy should have shape [B, L, 2]"
    assert ta_time_mask.dim() == 2 and ta_time_mask.shape[:2] == xy.shape[:2], \
        "ta_time_mask should have shape [B, L] and match xy batch/time dimensions"

    # Query the signed distance field (SDF): d > 0 = free space, d < 0 = inside obstacle
    d = sdf_fn(xy)  # [B, L]

    # Pointwise loss terms
    L_obs_pt = F.relu(-d) ** 2                      # Penalize penetration (d < 0)
    L_clear_pt = F.relu(margin - d) ** 2 if w_clear > 0 else torch.zeros_like(d)

    # Apply the time-step mask (only compute loss for imputed time steps)
    mask = ta_time_mask
    eps = 1e-8
    denom = mask.sum().clamp_min(eps)

    # Aggregate losses
    if reduction == "mean":
        #print(f"Denominator for mean reduction: {denom.item()}")
        L_obs = (L_obs_pt * mask).sum() / denom
        L_clear = (L_clear_pt * mask).sum() / denom
        path_penalty = (L_obs_pt * mask).max(dim=1).values.mean()
        loss = w_obs * L_obs + w_clear * L_clear
    elif reduction == "sum":
        L_obs = (L_obs_pt * mask).sum()
        L_clear = (L_clear_pt * mask).sum()
        path_penalty = (L_obs_pt * mask).max(dim=1).values.sum()
        loss = w_obs * L_obs + w_clear * L_clear
    elif reduction == "none":
        # Return element-wise tensors (mask applied)
        L_obs = L_obs_pt * mask
        L_clear = L_clear_pt * mask
        path_penalty = L_obs_pt * mask
        loss = w_obs * L_obs + w_clear * L_clear
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")

    ## Compute summary statistics (for logging / monitoring)
    #with torch.no_grad() if detach_stats else torch.enable_grad():
    #    collision_rate = ((d < 0).float() * mask).sum() / denom
    #    mean_clearance = (torch.clamp_min(d, 0.0) * mask).sum() / denom

    return {
        "loss": loss,
        "L_obs": L_obs,
        "L_clear": L_clear,
        "L_path": path_penalty,
    #    "collision_rate": collision_rate.detach() if detach_stats else collision_rate,
    #    "mean_clearance": mean_clearance.detach() if detach_stats else mean_clearance
    }


def compute_social_collision_loss(
    pred_xy: torch.Tensor,
    target_time_mask: torch.Tensor,
    neighbor_xy: torch.Tensor,
    neighbor_mask: torch.Tensor,
    *,
    margin: float = 0.04,
    path_weight: float = 1.0,
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    """
    Margin loss for ego-neighbor collision avoidance.

    pred_xy: [B, L, 2], normalized predicted ego trajectory.
    target_time_mask: [B, L], steps being optimized/evaluated.
    neighbor_xy: [B, N, L, 2], normalized neighbor trajectories.
    neighbor_mask: [B, N, L], valid neighbor states.
    """
    assert pred_xy.dim() == 3 and pred_xy.size(-1) == 2
    assert neighbor_xy.dim() == 4 and neighbor_xy.size(-1) == 2
    assert neighbor_mask.shape[:3] == neighbor_xy.shape[:3]

    rel = pred_xy.unsqueeze(1) - neighbor_xy
    dist = torch.linalg.norm(rel, dim=-1)  # [B, N, L]
    valid_neighbors = neighbor_mask.float() > 0
    valid_time = target_time_mask.float() > 0
    valid = valid_neighbors & valid_time.unsqueeze(1)

    # Use the closest valid neighbor at each time step. This preserves the
    # temporal safety signal instead of diluting it over all far-away agents.
    inf = torch.full_like(dist, float("inf"))
    nearest_dist = torch.where(valid, dist, inf).min(dim=1).values  # [B, L]
    has_neighbor = torch.isfinite(nearest_dist)
    nearest_dist = torch.where(has_neighbor, nearest_dist, torch.full_like(nearest_dist, margin))
    active_time = valid_time & has_neighbor

    # Normalize by the margin so a direct overlap has O(1) point loss.
    point_loss = (F.relu(margin - nearest_dist) / max(float(margin), 1e-8)) ** 2
    point_loss = point_loss * active_time.float()
    denom = active_time.float().sum().clamp_min(1.0)

    if reduction == "mean":
        point_mean = point_loss.sum() / denom
        # Differentiable trajectory-level penalty: any severe near-collision
        # raises the whole path loss, matching path-level collision metrics.
        path_loss = (1.0 - torch.exp(-point_loss.sum(dim=1))).mean()
        loss = point_mean + float(path_weight) * path_loss
    elif reduction == "sum":
        point_mean = point_loss.sum()
        path_loss = 1.0 - torch.exp(-point_loss.sum(dim=1))
        loss = point_mean + float(path_weight) * path_loss.sum()
    elif reduction == "none":
        point_mean = point_loss
        path_loss = 1.0 - torch.exp(-point_loss.sum(dim=1))
        loss = point_loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")

    with torch.no_grad():
        collision_rate = (((dist < margin) & valid).any(dim=1).float() * valid_time).sum() / valid_time.sum().clamp_min(1.0)
        collision_rate = collision_rate.detach()

    return {
        "loss": loss,
        "L_social": loss,
        "L_social_point": point_mean,
        "L_path": path_loss,
        "social_collision_rate": collision_rate,
    }
