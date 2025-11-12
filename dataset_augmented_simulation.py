import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from torch.utils.data import Sampler

import random
from dataset_simulation import Simulation_Dataset
from dataset_simulation import ScenarioBatchDataLoader
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple
try:
    # Optional import; only used when preprocess_for_resnet=True
    from torchvision.models import ResNet18_Weights
except Exception:
    ResNet18_Weights = None

def resize_map(map_img, target_shape):
    # cv2 expects (width, height)
    target_shape_cv2 = (target_shape[1], target_shape[0])
    return cv2.resize(map_img, target_shape_cv2, interpolation=cv2.INTER_LINEAR)

# --- AugmentedSimulationDataset ---

class AugmentedSimulationDataset(Simulation_Dataset):
    def __init__(
        self,
        *args,
        augmentation_mode='crop_and_resize',
        resize_to=(128, 128),
        debug=False,
        preprocess_for_resnet=False,
        scen_map_variant: str = "default",  # "default" or "merged_last_empty_with_poi"
        poi_radius: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.augmentation_mode = augmentation_mode
        self.default_resize_to = resize_to # (height, width), this is not used if augmentation_mode is None
        self.debug = debug
        # If True, convert scen_map to RGB float in [0,1] and normalize with ImageNet stats
        self.preprocess_for_resnet = preprocess_for_resnet
        # Variant of scen_map fed to model (raw stays unchanged for evaluation)
        self.scen_map_variant = scen_map_variant
        self.poi_radius = int(poi_radius)
        if not self.load_scenario_map:
            raise ValueError("AugmentedSimulationDataset requires load_scenario_map=True")
        # Cache ImageNet mean/std from torchvision weights if available
        if self.preprocess_for_resnet:
            # Resolve ImageNet mean/std robustly across torchvision versions
            def _fallback_stats():
                print("⚠️ Warning: Using fallback ImageNet mean/std values.")
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
                return mean, std

            if ResNet18_Weights is None:
                self._imnet_mean, self._imnet_std = _fallback_stats()
            else:
                try:
                    w = ResNet18_Weights.DEFAULT
                    mean = std = None
                    meta = getattr(w, "meta", None)
                    if isinstance(meta, dict):
                        mean = meta.get("mean", None)
                        std = meta.get("std", None)
                    if mean is None or std is None:
                        try:
                            t = w.transforms()
                            mean = getattr(t, "mean", mean)
                            std = getattr(t, "std", std)
                        except Exception:
                            pass
                    if mean is None or std is None:
                        print("⚠️ Warning: Cannot find ImageNet mean/std from torchvision; using fallback values.")
                        self._imnet_mean, self._imnet_std = _fallback_stats()
                    else:
                        print(f"Using ImageNet mean: {mean}, std: {std} for ResNet preprocessing")
                        self._imnet_mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
                        self._imnet_std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
                except Exception:
                    print("⚠️ Warning: Exception when loading torchvision weights; using fallback ImageNet mean/std.")
                    self._imnet_mean, self._imnet_std = _fallback_stats()

    def __getitem__(self, key):
        if isinstance(key, tuple):
            index, batch_params = key
            resize_to = batch_params.resize_to
        else:
            index, batch_params = key, None
            resize_to = self.default_resize_to

        index, scenario = self.get_index(index)
        track = self.data[scenario][index]

        observed_values, observed_masks, gt_masks, time_points, person_ids = self._parse_single_data(
            track, scenario, missing_ratio=self.missing_ratio, missing_strategy=self.missing_strategy)

        scen_map_raw = self.scen_map[scenario]
        desired_crop_size = resize_to

        # Draw the track on the scenario map for debugging
        if self.debug:
            scen_map_raw = scen_map_raw.copy()
            map_height = scen_map_raw.shape[0]
            for t in range(len(observed_values)):
                x, y = observed_values[t]
                y = y * self.scen_map_base_scale  # Scale y-coordinate by the base scale
                x = x * self.scen_map_base_scale  # Scale x-coordinate by the base
                # Flip y-coordinate to match image coordinates
                y = map_height - y

                if 0 <= x < scen_map_raw.shape[1] and 0 <= y < scen_map_raw.shape[0]:
                    cv2.circle(scen_map_raw, (int(x), int(y)), 13, (0, 0, 255), -1)

        if self.augmentation_mode == 'crop_and_resize':
            # Use maybe_augment_sample to get the processed scenario map and adjusted track data
            scen_map_processed, adjusted_values, ppw = self.maybe_augment_sample(
                scen_map_raw, observed_values, observed_masks, desired_crop_size
            )
            # Use actual shape of the processed scenario map
            resize_shape = resize_to
        elif self.augmentation_mode == 'pad_and_resize':
            final_map, adjusted_values, ppw = self.pad_and_resize_with_track(scen_map_raw, observed_values, desired_crop_size)
            scen_map_processed = final_map
            resize_shape = resize_to
            #return final_map, adjusted_values, scale # 1.0 scale means no scaling applied
        elif self.augmentation_mode == None:
            # No augmentation: per-axis pixels-per-world
            ppw = np.array([self.scen_map_base_scale, self.scen_map_base_scale], dtype=np.float32)
            # convert the coordinates according to the scenario map size
            adjusted_values = self.convert_to_track_coordinates(observed_values, scen_map_raw.shape[0])
            scen_map_processed = scen_map_raw
            resize_shape = (-1, -1) # Indicate no resizing applied
        
        # Convert pixels-per-world (ppw) to world-per-pixel (wpp) per-axis for downstream use
        # Guard against division by zero
        ppw = np.array(ppw, dtype=np.float32)
        wpp = 1.0 / np.clip(ppw, 1e-8, None)

        # Normalize adjusted track coordinates to [0, 1] based on final scenario map shape
        map_height, map_width = scen_map_processed.shape[:2]
        map_size = np.array([map_width, map_height], dtype=np.float32)
        adjusted_values = adjusted_values / map_size  # shape (T, 2)


        # Optional: preprocess scenario map for torchvision ResNet
        # Expected by pretrained ResNet: RGB, float32 in [0,1], then ImageNet mean/std normalization
        # Apply alternative scenemap variant if requested (on processed map only)
        # Keep scen_map_raw unchanged for evaluation.
        adjusted_values_px = None
        if self.scen_map_variant and self.scen_map_variant != "default":
            # Merge channels 0 and 1 into channel 0; move entrances (2) to channel 1; clear channel 2
            base = scen_map_processed
            merged = np.maximum(base[:, :, 0], base[:, :, 1])
            variant = np.zeros_like(base)
            variant[:, :, 0] = merged
            variant[:, :, 1] = base[:, :, 2]
            # Draw start/end markers as Gaussians on channel 2
            # Map normalized coords back to pixel coords if needed
            # At this point adjusted_values may have been normalized already (see above).
            # Recover pixel coords using the current map size.
            adjusted_values_px = adjusted_values * np.array([map_width, map_height], dtype=np.float32)
            self._draw_start_end_gaussians(variant, adjusted_values_px, observed_masks, self.poi_radius)
            scen_map_processed = variant

        if self.preprocess_for_resnet:
            # Keep a raw copy (BGR uint8/float in [0,255]) for evaluation/collision metrics
            scen_map_raw = scen_map_processed.copy() if self.scen_map_variant == "default" else base.copy()
            img = scen_map_processed
            # Ensure float32 and [0,1]
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            # ImageNet normalization
            img = (img - self._imnet_mean) / self._imnet_std
            scen_map_processed = img

        s = {
            'observed_data': adjusted_values,
            'observed_mask': observed_masks,
            'gt_mask': gt_masks,
            'timepoints': np.arange(self.data_length),
            'person_ids': person_ids,
            'scen_map': scen_map_processed,
            'scen_map_scale': wpp.astype(np.float32),
            'resize_shape': resize_shape
        }
        if self.preprocess_for_resnet:
            # Provide raw RGB map for downstream evaluation (collision metrics)
            s['scen_map_raw'] = scen_map_raw
        return s

    def _draw_start_end_gaussians(self, img: np.ndarray, track_px: np.ndarray, masks: np.ndarray, radius: int = 5):
        """
        Draw two Gaussian blobs (start and end positions) onto channel 2 of img.
        - img: HxWx3 uint8
        - track_px: Tx2 float pixel coordinates aligned with img
        - masks: Tx2 boolean masks; True where observed
        - radius: Gaussian radius in pixels
        """
        H, W = img.shape[:2]
        if track_px is None or len(track_px) == 0:
            return
        # Determine first/last valid indices
        valid = np.any(masks, axis=1)
        if not np.any(valid):
            # Fallback to endpoints
            idx_start, idx_end = 0, len(track_px) - 1
        else:
            idxs = np.where(valid)[0]
            idx_start, idx_end = int(idxs[0]), int(idxs[-1])

        centers = [track_px[idx_start], track_px[idx_end]]

        # Precompute Gaussian kernel (normalized to 255)
        R = max(1, int(radius))
        size = 2 * R + 1
        yy, xx = np.ogrid[-R:R+1, -R:R+1]
        sigma = max(1e-6, R / 2.0)
        kernel = np.exp(-(xx*xx + yy*yy) / (2 * sigma * sigma))
        kernel = (kernel / kernel.max()) * 255.0
        kernel = kernel.astype(np.uint8)

        ch = img[:, :, 2]
        for c in centers:
            x, y = int(round(c[0])), int(round(c[1]))
            if x < 0 or x >= W or y < 0 or y >= H:
                continue
            x0, y0 = x - R, y - R
            x1, y1 = x + R + 1, y + R + 1
            kx0, ky0 = 0, 0
            kx1, ky1 = size, size
            # Clip to image bounds
            if x0 < 0:
                kx0 = -x0
                x0 = 0
            if y0 < 0:
                ky0 = -y0
                y0 = 0
            if x1 > W:
                kx1 -= (x1 - W)
                x1 = W
            if y1 > H:
                ky1 -= (y1 - H)
                y1 = H
            if x0 >= x1 or y0 >= y1:
                continue
            roi = ch[y0:y1, x0:x1]
            kroi = kernel[ky0:ky1, kx0:kx1]
            # Blend by max to preserve bright signal
            np.maximum(roi, kroi, out=roi)
        img[:, :, 2] = ch

    def crop_map_and_adjust_track(self, scen_map_raw, observed_values, observed_masks, desired_aspect, min_bbox_ratio=0.15):
        H, W, _ = scen_map_raw.shape
        masked_value = np.where(observed_masks, observed_values, np.nan)
        min_x, min_y = np.nanmin(masked_value, axis=0)
        max_x, max_y = np.nanmax(masked_value, axis=0)

        min_x = np.floor(min_x)
        max_x = np.ceil(max_x)
        min_y = np.floor(min_y)
        max_y = np.ceil(max_y)
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        # Insert warning for invalid bbox (silenced unless debug=True)
        if bbox_w <= 0 or bbox_h <= 0:
            if self.debug:
                print("Warning: Invalid bounding box dimensions.")
                print(f"min_x = {min_x}, max_x = {max_x}, bbox_w = {bbox_w}")
                print(f"min_y = {min_y}, max_y = {max_y}, bbox_h = {bbox_h}")

        # Ensure bbox is not too small compared to image size (using min_bbox_ratio)
        min_bbox_w = int(min_bbox_ratio * W)
        min_bbox_h = int(min_bbox_ratio * H)

        if bbox_w < min_bbox_w:
            diff = min_bbox_w - bbox_w
            min_x -= diff / 2
            max_x += diff / 2
            bbox_w = max_x - min_x
            #print(f"[Adjust bbox] Expanded bbox width to minimum {min_bbox_w:.1f}")

        if bbox_h < min_bbox_h:
            diff = min_bbox_h - bbox_h
            min_y -= diff / 2
            max_y += diff / 2
            bbox_h = max_y - min_y
            #print(f"[Adjust bbox] Expanded bbox height to minimum {min_bbox_h:.1f}")

        scale_param = 10
        margin_left_limit = min(int(scale_param * bbox_w), int(min_x))
        margin_right_limit = min(int(scale_param * bbox_w), int(W - max_x))
        margin_top_limit = min(int(scale_param * bbox_h), int(min_y))
        margin_bottom_limit = min(int(scale_param * bbox_h), int(H - max_y))

        ## Debug only if any of the ranges are negative
        #if margin_left_limit < 0 or margin_right_limit < 0 or margin_top_limit < 0 or margin_bottom_limit < 0:
        #    print("[DEBUG] Potential invalid randint range detected:")
        #    print(f"  scale_param * bbox_w = {scale_param * bbox_w}")
        #    print(f"  scale_param * bbox_h = {scale_param * bbox_h}")
        #    print(f"  min_x = {min_x}, W - max_x = {W - max_x}")
        #    print(f"  min_y = {min_y}, H - max_y = {H - max_y}")
        #    print(f"  margin_left_limit = {margin_left_limit}, margin_right_limit = {margin_right_limit}")
        #    print(f"  margin_top_limit = {margin_top_limit}, margin_bottom_limit = {margin_bottom_limit}")

        margin_left = random.randint(0, max(0, margin_left_limit))
        margin_right = random.randint(0, max(0, margin_right_limit))
        margin_top = random.randint(0, max(0, margin_top_limit))
        margin_bottom = random.randint(0, max(0, margin_bottom_limit))

        target_aspect = desired_aspect
        current_aspect = (max_x - min_x + margin_left) / (max_y - min_y + margin_top)
        if current_aspect > target_aspect:
            margin_bottom = int((max_x - min_x + margin_left + margin_right) / target_aspect - (max_y - min_y + margin_top))
        else:
            margin_right = int((max_y - min_y + margin_top + margin_bottom) * target_aspect - (max_x - min_x + margin_left))

        left = max(int(min_x) - margin_left, 0)
        right = min(int(max_x) + margin_right, W)
        top = max(int(min_y) - margin_top, 0)
        bottom = min(int(max_y) + margin_bottom, H)

        # Insert validation before return
        if bottom <= top or right <= left:
            raise ValueError(f"Invalid crop box: top={top}, bottom={bottom}, left={left}, right={right}")

        cropped_map = scen_map_raw[top:bottom, left:right]
        adjusted_values = observed_values - np.array([left, top])
        return cropped_map, adjusted_values, (top, left)

    def pad_to_aspect_ratio(self, img, track, desired_aspect, pad_value=(0, 255, 0)):
        h, w = img.shape[:2]
        current_aspect = w / h if h != 0 else 1.0

        pad_top = pad_bottom = pad_left = pad_right = 0
        if current_aspect > desired_aspect:
            target_h = int(w / desired_aspect)
            pad_total = max(target_h - h, 0)
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
        elif current_aspect < desired_aspect:
            target_w = int(h * desired_aspect)
            pad_total = max(target_w - w, 0)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_value)
            track[:, 0] += pad_left
            track[:, 1] += pad_top
        return img, track

    def resize_map_and_track(self, img, track, target_size):
        if img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError("Cannot resize image with zero dimension.")
        scale_x = target_size[1] / img.shape[1]
        scale_y = target_size[0] / img.shape[0]
        track[:, 0] *= scale_x
        track[:, 1] *= scale_y
        # Check for NaN or Inf in track
        if np.isnan(track).any() or np.isinf(track).any():
            raise ValueError("NaN or Inf detected in track after resizing.")
        img_resized = cv2.resize(img, target_size[::-1], interpolation=cv2.INTER_LINEAR)
        return img_resized, track, (scale_x, scale_y)

    def random_crop_and_resize_with_track(self, scen_map_raw, observed_values, observed_masks, desired_crop_size):
        observed_values = self.convert_to_track_coordinates(observed_values, scen_map_raw.shape[0])
        aspect = desired_crop_size[1] / desired_crop_size[0]
        cropped_map, track_adj, _ = self.crop_map_and_adjust_track(scen_map_raw, observed_values, observed_masks, aspect)
        padded_map, track_padded = self.pad_to_aspect_ratio(cropped_map, track_adj, aspect)
        final_map, track_resized, (scale_x, scale_y) = self.resize_map_and_track(padded_map, track_padded, desired_crop_size)

        if abs(scale_x - scale_y) > 1e-3:
            print(f"Warning: Non-uniform scaling detected: scale_x={scale_x}, scale_y={scale_y}")

        # Insert NaN/Inf and empty image checks before return
        if np.isnan(track_resized).any() or np.isinf(track_resized).any():
            raise ValueError("NaN or Inf detected in resized track!")
        if final_map.size == 0:
            raise ValueError("Empty image after crop!")

        # Compute per-axis pixels-per-world scale after augmentation
        ppw_x = self.scen_map_base_scale * scale_x
        ppw_y = self.scen_map_base_scale * scale_y
        final_scale = np.array([ppw_x, ppw_y], dtype=np.float32)
        # Insert scale range check
        if (
            np.any(final_scale < 1e-6)
            or np.any(final_scale > 1e6)
            or np.isnan(final_scale).any()
            or np.isinf(final_scale).any()
        ):
            print(f"⚠️ Rejected sample due to extreme scale: {final_scale}")
            print(f"[DEBUG] scen_map_raw.shape = {scen_map_raw.shape}")
            print(f"[DEBUG] observed_values (scaled and flipped):\n{observed_values}")
            print(f"[DEBUG] track_resized:\n{track_resized}")
            print(f"[DEBUG] scale_x = {scale_x}, scale_y = {scale_y}")
            print(f"[DEBUG] final_map.shape = {final_map.shape}")
        return final_map, track_resized, final_scale

    def convert_to_track_coordinates(self, observed_values, H):
        """
        Convert observed values to track coordinates by transfer from left-down to right-down coordinates and scale it to the scenario map scale.
        """
        adjusted_values = observed_values.copy() * self.scen_map_base_scale
        adjusted_values[:, 1] = H - adjusted_values[:, 1]
        return adjusted_values

    def maybe_augment_sample(self, scen_map_raw, observed_values, observed_masks, desired_crop_size):
        """
        With 50% probability, apply random_crop_and_resize_with_track; otherwise pad+resize.
        Returns per-axis pixels-per-world (ppw) scales reflecting augmentation.
        """
        if random.random() < 0.5:
            #observed_values = self.convert_to_track_coordinates(observed_values, scen_map_raw.shape[0])
            final_map, adjusted_values, scale = self.pad_and_resize_with_track(scen_map_raw, observed_values, desired_crop_size)

            return final_map, adjusted_values, scale # 1.0 scale means no scaling applied
        else:
            final_map, adjusted_values, scale = self.random_crop_and_resize_with_track(scen_map_raw, observed_values, observed_masks, desired_crop_size)
            return final_map, adjusted_values, scale

    def pad_and_resize_with_track(self, scen_map_raw, observed_values, desired_crop_size):
        """
        Pads the scenario map to the target aspect ratio and resizes it,
        adjusting track coordinates accordingly.
        desired_crop_size: (height, width)
        """
        # Convert coordinates
        observed_values = self.convert_to_track_coordinates(observed_values, scen_map_raw.shape[0])

        # Pad to aspect ratio
        aspect = desired_crop_size[1] / desired_crop_size[0] # width/height
        padded_map, track_padded = self.pad_to_aspect_ratio(scen_map_raw, observed_values.copy(), aspect)

        # Resize
        final_map, track_resized, (scale_x, scale_y) = self.resize_map_and_track(padded_map, track_padded, desired_crop_size)

        if scale_x != scale_y:
            print(f"Warning: Non-uniform scaling in pad_and_resize_with_track: scale_x={scale_x}, scale_y={scale_y}")

        # Insert NaN/Inf and empty image checks before return
        if np.isnan(track_resized).any() or np.isinf(track_resized).any():
            raise ValueError("NaN or Inf detected in track after pad and resize!")
        if final_map.size == 0:
            raise ValueError("Empty image after pad and resize!")

        # Compute per-axis pixels-per-world scale after augmentation
        ppw_x = self.scen_map_base_scale * scale_x
        ppw_y = self.scen_map_base_scale * scale_y
        final_scale = np.array([ppw_x, ppw_y], dtype=np.float32)
        # Insert scale range check for pad_and_resize_with_track
        if (
            np.any(final_scale < 1e-6)
            or np.any(final_scale > 1e6)
            or np.isnan(final_scale).any()
            or np.isinf(final_scale).any()
        ):
            print(f"⚠️ Rejected padded sample due to extreme scale: {final_scale}")
            #raise ValueError(f"Extreme final scale in pad and resize: {final_scale}")
        return final_map, track_resized, final_scale


def get_augmented_dataloader_old(
    data_length,
    seed,
    scenarios=None,
    batch_size=8,
    load_scenario_map=True,
    zero_based_position=True,
    resize_options=[(384, 384), (512, 512), (768, 768)], # List of (height, width) tuples
    augmentation_mode='crop_and_resize',
    part="train",
    debug=False,
    preprocess_for_resnet=True,
    scen_map_variant="default",
    poi_radius=5,
):
    from torch.utils.data import DataLoader
    assert part in ["train", "valid", "test"], "part must be 'train',  'valid' or 'test'"
    def build_loader(mode):
        base_dataset = AugmentedSimulationDataset(
            data_length=data_length,
            subset_split_seed=seed,
            scenarios=scenarios,
            mode=mode,
            load_scenario_map=load_scenario_map,
            zero_based_position=zero_based_position,
            augmentation_mode=augmentation_mode,
            preprocess_for_resnet=preprocess_for_resnet,
            debug=debug,
            scen_map_variant=scen_map_variant,
            poi_radius=poi_radius,
        )
        wrapped_dataset = ResizeWrapperDataset(base_dataset, resize_options, batch_size)
        return DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=(mode == 'train'))
    if augmentation_mode is None:
        print("⚠️ Warning: augmentation_mode is None, no augmentation will be applied and there will error. You may want to use get_nonaugmented_dataloader_with_scenario_batches instead for better performance.")
    
    return build_loader(part)

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler

def get_augmented_dataloader(
    data_length,
    seed,
    scenarios=None,
    batch_size=8,
    load_scenario_map=True,
    zero_based_position=True,
    resize_options=((384,384), (512,512), (768,768)),
    augmentation_mode='crop_and_resize',
    part="train",
    debug=False,
    distributed=False,
    drop_last=True,
    num_workers=8,
    preprocess_for_resnet=True,
    scen_map_variant="default",
    poi_radius=5,
):
    assert part in ["train", "valid", "test", "unit_test"], "part must be 'train', 'valid', 'test' or 'unit_test'"
    base_dataset = AugmentedSimulationDataset(
        data_length=data_length,
        subset_split_seed=seed,
        scenarios=scenarios,
        mode=part,
        load_scenario_map=load_scenario_map,
        zero_based_position=zero_based_position,
        augmentation_mode=augmentation_mode,
        resize_to=resize_options[0],  # 只是默认值；真正按批的来自 sampler
        preprocess_for_resnet=preprocess_for_resnet,
        debug=debug,
        scen_map_variant=scen_map_variant,
        poi_radius=poi_radius,
    )

    if distributed:
        base_sampler = DistributedSampler(base_dataset, shuffle=(part=="train"), seed=seed, drop_last=drop_last)
    else:
        base_sampler = RandomSampler(base_dataset) if part=="train" else torch.utils.data.SequentialSampler(base_dataset)

    batch_sampler = BatchParamSampler(
        base_sampler=base_sampler,
        batch_size=batch_size,
        resize_options=resize_options,
        drop_last=drop_last,
        seed=seed,
    )

    loader = DataLoader(
        base_dataset,
        batch_sampler=batch_sampler,   # ⚠️ 不要再传 batch_size/shuffle
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers>0),
        collate_fn=None,               # 默认就好（或写个轻量的，用于 pad 轨迹）
        prefetch_factor=4 if num_workers>0 else None,
    )
    return loader, batch_sampler  # 训练循环里每个 epoch 记得 set_epoch


# Utility function to get a non-augmented dataloader using scenario batches
def get_nonaugmented_dataloader_with_scenario_batches(
    data_length,
    seed,
    scenarios=None,
    batch_size=8,
    load_scenario_map=True,
    zero_based_position=True,
    debug=False,
    part="train",
    preprocess_for_resnet=True,
    scen_map_variant="default",
    poi_radius=5,
):
    assert part in ["train", "valid", "test", "unit_test"], "part must be 'train', 'valid', 'unit_test' or 'test'"
    def build_loader(mode):
        dataset = AugmentedSimulationDataset(
            data_length=data_length,
            subset_split_seed=seed,
            scenarios=scenarios,
            mode=mode,
            load_scenario_map=load_scenario_map,
            zero_based_position=zero_based_position,
            augmentation_mode=None,  # No augmentation applied
            preprocess_for_resnet=preprocess_for_resnet,
            debug=debug,
            scen_map_variant=scen_map_variant,
            poi_radius=poi_radius,
        )
        return ScenarioBatchDataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'))
    return build_loader(part)


@dataclass(frozen=True)
class BatchParams:
    resize_to: Tuple[int, int]     # (H, W)

class BatchParamSampler(Sampler):
    def __init__(self, base_sampler: Sampler, batch_size: int, resize_options, drop_last=False, seed=0):
        self.base_sampler = base_sampler
        self.batch_size = batch_size
        self.resize_options = tuple(resize_options)
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if hasattr(self.base_sampler, "set_epoch"):
            self.base_sampler.set_epoch(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batch = []
        for idx in self.base_sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                params = BatchParams(resize_to=rng.choice(self.resize_options))
                yield [(i, params) for i in batch]
                batch = []
        if len(batch) and not self.drop_last:
            params = BatchParams(resize_to=rng.choice(self.resize_options))
            yield [(i, params) for i in batch]

    def __len__(self):
        n = len(self.base_sampler)
        return (n // self.batch_size) if self.drop_last else math.ceil(n / self.batch_size)

from torch.utils.data._utils.collate import default_collate
import numpy as np
import torch

def debug_collate(batch):
    print("---- DEBUG COLLATE ----")
    for k in batch[0].keys():
        types = []
        shapes = []
        dtypes = []
        for i, b in enumerate(batch):
            v = b[k]
            if isinstance(v, np.ndarray):
                types.append("np")
                shapes.append(v.shape)
                dtypes.append(v.dtype)
            elif torch.is_tensor(v):
                types.append("torch")
                shapes.append(tuple(v.shape))
                dtypes.append(v.dtype)
            else:
                types.append(type(v).__name__)
                shapes.append(getattr(v, "shape", None))
                dtypes.append(getattr(v, "dtype", None))
        print(f"{k:>16s} | types={set(types)} dtypes={set(map(str,dtypes))} shapes={set(map(str,shapes))}")
    print("-----------------------")
    return default_collate(batch)