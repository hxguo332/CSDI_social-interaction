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
        loss = w_obs * L_obs + w_clear * L_clear
    elif reduction == "sum":
        L_obs = (L_obs_pt * mask).sum()
        L_clear = (L_clear_pt * mask).sum()
        loss = w_obs * L_obs + w_clear * L_clear
    elif reduction == "none":
        # Return element-wise tensors (mask applied)
        L_obs = L_obs_pt * mask
        L_clear = L_clear_pt * mask
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
    dist = torch.linalg.norm(rel, dim=-1)
    valid = neighbor_mask.float() * target_time_mask.unsqueeze(1).float()
    point_loss = F.relu(margin - dist) ** 2 * valid
    denom = valid.sum().clamp_min(1e-8)

    if reduction == "mean":
        loss = point_loss.sum() / denom
    elif reduction == "sum":
        loss = point_loss.sum()
    elif reduction == "none":
        loss = point_loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")

    with torch.no_grad():
        collision_rate = (((dist < margin).float() * valid).sum() / denom).detach()

    return {
        "loss": loss,
        "L_social": loss,
        "social_collision_rate": collision_rate,
    }
