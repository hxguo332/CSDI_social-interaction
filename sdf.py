import numpy as np
from typing import Optional, Union

def generate_sdf(
    scenario_map_raw: Union[np.ndarray, "torch.Tensor"],
    *,
    obstacle_threshold: int = 255,
    nogo_threshold: int = 255,
    pixel_size: Optional[float] = None,
) -> np.ndarray:
    """
    Build a signed distance field (SDF) from an RGB scenario map.

    Inputs
    ------
    scenario_map_raw : np.ndarray or torch.Tensor
        RGB map. Expected shapes:
          - (H, W, 3)  [channels-last], or
          - (3, H, W)  [channels-first]
        Dtype can be uint8/float; values are binarized as below.
        Channel semantics:
          - channel 0 (R): obstacles (non-zero by default or via obstacle_threshold)
          - channel 1 (G): other no-go area (value == 255 is active region)

    obstacle_threshold : int, default=1
        Pixels in channel-0 with value >= obstacle_threshold are considered obstacle.

    nogo_threshold : int, default=255
        Pixels in channel-1 with value >= this value are considered no-go (obstacle).

    pixel_size : float or None, default=None
        If provided, EDT distances are scaled by this factor (e.g., meters/pixel).
        If None, distances are in pixel units.

    Returns
    -------
    sdf : np.ndarray, shape (H, W), dtype float32
        Signed distance field:
          +d in free space (distance to nearest obstacle),
          -d inside obstacles/no-go (negative distance to free).
    """
    # Lazy import to avoid hard dependency if caller doesn't use this path
    try:
        from scipy.ndimage import distance_transform_edt as edt
    except Exception as e:
        raise ImportError("SciPy is required: pip install scipy") from e

    # Convert to numpy (H, W, 3), uint8
    if "torch" in str(type(scenario_map_raw)):
        scenario_map_raw = scenario_map_raw.detach().cpu().numpy()

    arr = scenario_map_raw
    if arr.ndim != 3 or (arr.shape[2] != 3 and arr.shape[0] != 3):
        raise ValueError("Expected RGB array with shape (H,W,3) or (3,H,W).")

    if arr.shape[0] == 3 and arr.shape[2] != 3:  # channels-first -> channels-last
        arr = np.transpose(arr, (1, 2, 0))

    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        # Normalize if floats in [0,1]; otherwise cast conservatively
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    H, W, _ = arr.shape
    ch0 = arr[..., 0]  # obstacles
    ch1 = arr[..., 1]  # other no-go (255 active)

    # Binary obstacle mask: 1 = obstacle/no-go, 0 = free
    mask_obst = (ch0 >= obstacle_threshold) | (ch1 >= nogo_threshold)
    mask_obst = mask_obst.astype(np.uint8)

    # EDT:
    #  dist_to_obst: distance from free pixels to nearest obstacle pixel
    #  dist_to_free: distance from obstacle pixels to nearest free pixel
    # SciPy's edt treats zeros as the "features": so we pass complements accordingly.
    if pixel_size is None:
        dist_to_obst = edt(1 - mask_obst)  # free -> nearest obstacle
        dist_to_free = edt(mask_obst)      # obstacle -> nearest free
    else:
        # scale distances into world units directly
        dist_to_obst = edt(1 - mask_obst, sampling=pixel_size)
        dist_to_free = edt(mask_obst,     sampling=pixel_size)

    #sdf = dist_to_obst - dist_to_free  # +free, -inside
    H, W = mask_obst.shape
    sdf = (dist_to_obst - dist_to_free) / np.sqrt(H**2 + W**2) # normalize to [−1,1] range
    return sdf.astype(np.float32)