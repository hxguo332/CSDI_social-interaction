import matplotlib.pyplot as plt
import numpy as np

import random
from dataset_simulation import Simulation_Dataset
import cv2
import numpy as np

def resize_map(map_img, target_shape):
    # cv2 expects (width, height)
    target_shape_cv2 = (target_shape[1], target_shape[0])
    return cv2.resize(map_img, target_shape_cv2, interpolation=cv2.INTER_LINEAR)

# --- AugmentedSimulationDataset ---

class AugmentedSimulationDataset(Simulation_Dataset):
    def __init__(self, *args, augmentation_mode='crop_and_resize', resize_to=(128, 128), debug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_mode = augmentation_mode
        self.resize_to = resize_to
        self.debug = debug
        if not self.load_scenario_map:
            raise ValueError("AugmentedSimulationDataset requires load_scenario_map=True")

    def __getitem__(self, index):
        index, scenario = self.get_index(index)
        track = self.data[scenario][index]

        observed_values, observed_masks, gt_masks, time_points, person_ids = self._parse_single_data(
            track, scenario, missing_ratio=self.missing_ratio, missing_strategy=self.missing_strategy)

        scen_map_raw = self.scen_map[scenario]
        desired_crop_size = self.resize_to

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

        # Use maybe_augment_sample to get the processed scenario map and adjusted track data
        scen_map_processed, adjusted_values, rescale = self.maybe_augment_sample(
            scen_map_raw, observed_values, observed_masks, desired_crop_size
        )

        s = {
            'observed_data': adjusted_values,
            'observed_mask': observed_masks,
            'gt_mask': gt_masks,
            'timepoints': np.arange(self.data_length),
            'person_ids': person_ids,
            'scen_map': scen_map_processed,
            'scen_map_scale': np.full(2, rescale),
            'resize_shape': self.resize_to
        }
        return s

    def crop_map_and_adjust_track(self, scen_map_raw, observed_values, observed_masks, desired_aspect, min_size=30):
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
        # Insert warning for invalid bbox
        if bbox_w <= 0 or bbox_h <= 0:
            print("Warning: Invalid bounding box dimensions.")
            print(f"min_x = {min_x}, max_x = {max_x}, bbox_w = {bbox_w}")
            print(f"min_y = {min_y}, max_y = {max_y}, bbox_h = {bbox_h}")

        if bbox_w < min_size:
            min_x -= max(0, (min_size - bbox_w) / 2)
            max_x += min(W, (min_size - bbox_w) / 2)
            bbox_w = max_x - min_x
        if bbox_h < min_size:
            min_y -= max(0, (min_size - bbox_h) / 2)
            max_y += min(H, (min_size - bbox_h) / 2)
            bbox_h = max_y - min_y

        scale_param = 10
        margin_left_limit = min(int(scale_param * bbox_w), int(min_x))
        margin_right_limit = min(int(scale_param * bbox_w), int(W - max_x))
        margin_top_limit = min(int(scale_param * bbox_h), int(min_y))
        margin_bottom_limit = min(int(scale_param * bbox_h), int(H - max_y))

        # Debug only if any of the ranges are negative
        if margin_left_limit < 0 or margin_right_limit < 0 or margin_top_limit < 0 or margin_bottom_limit < 0:
            print("[DEBUG] Potential invalid randint range detected:")
            print(f"  scale_param * bbox_w = {scale_param * bbox_w}")
            print(f"  scale_param * bbox_h = {scale_param * bbox_h}")
            print(f"  min_x = {min_x}, W - max_x = {W - max_x}")
            print(f"  min_y = {min_y}, H - max_y = {H - max_y}")
            print(f"  margin_left_limit = {margin_left_limit}, margin_right_limit = {margin_right_limit}")
            print(f"  margin_top_limit = {margin_top_limit}, margin_bottom_limit = {margin_bottom_limit}")

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
        scale_x = target_size[1] / img.shape[1]
        scale_y = target_size[0] / img.shape[0]
        track[:, 0] *= scale_x
        track[:, 1] *= scale_y
        img_resized = cv2.resize(img, target_size[::-1], interpolation=cv2.INTER_LINEAR)
        return img_resized, track, (scale_x, scale_y)

    def random_crop_and_resize_with_track(self, scen_map_raw, observed_values, observed_masks, desired_crop_size):
        observed_values = self.convert_to_track_coordinates(observed_values, scen_map_raw.shape[0])
        aspect = desired_crop_size[1] / desired_crop_size[0]
        cropped_map, track_adj, _ = self.crop_map_and_adjust_track(scen_map_raw, observed_values, observed_masks, aspect)
        padded_map, track_padded = self.pad_to_aspect_ratio(cropped_map, track_adj, aspect)
        final_map, track_resized, (scale_x, scale_y) = self.resize_map_and_track(padded_map, track_padded, desired_crop_size)

        if scale_x != scale_y:
            print(f"Warning: Non-uniform scaling detected: scale_x={scale_x}, scale_y={scale_y}")

        final_scale = self.scen_map_base_scale * (scale_x + scale_y) / 2
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
        With 50% probability, apply nandom_crop_and_resize_with_track, otherwise return original data.
        The retured scale here means the scale applied to the track coordinates. For example, if the coordinates is 10 times larger than the original, the scale will be 10.
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
        """
        # Convert coordinates
        observed_values = self.convert_to_track_coordinates(observed_values, scen_map_raw.shape[0])

        # Pad to aspect ratio
        aspect = desired_crop_size[1] / desired_crop_size[0]
        padded_map, track_padded = self.pad_to_aspect_ratio(scen_map_raw, observed_values.copy(), aspect)

        # Resize
        final_map, track_resized, (scale_x, scale_y) = self.resize_map_and_track(padded_map, track_padded, desired_crop_size)

        if scale_x != scale_y:
            print(f"Warning: Non-uniform scaling in pad_and_resize_with_track: scale_x={scale_x}, scale_y={scale_y}")

        final_scale = self.scen_map_base_scale * (scale_x + scale_y) / 2
        return final_map, track_resized, final_scale

class ResizeWrapperDataset:
    def __init__(self, base_dataset, resize_options):
        self.base_dataset = base_dataset
        self.resize_options = resize_options
        self.current_resize_size = random.choice(self.resize_options)

    def __getitem__(self, index):
        self.base_dataset.resize_to = self.current_resize_size
        return self.base_dataset[index]

    def __len__(self):
        return len(self.base_dataset)

def get_augmented_dataloader(
    data_length,
    seed,
    scenarios=None,
    batch_size=8,
    load_scenario_map=True,
    zero_based_position=True,
    resize_options=[(384, 384), (512, 512), (768, 768)],
    augmentation_mode='crop_and_resize',
    debug=False
):
    from torch.utils.data import DataLoader
    def build_loader(mode):
        base_dataset = AugmentedSimulationDataset(
            data_length=data_length,
            subset_split_seed=seed,
            scenarios=scenarios,
            mode=mode,
            load_scenario_map=load_scenario_map,
            zero_based_position=zero_based_position,
            augmentation_mode=augmentation_mode,
            debug=debug
        )
        wrapped_dataset = ResizeWrapperDataset(base_dataset, resize_options=resize_options)
        return DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=(mode == 'train'))

    train_loader = build_loader("train")
    valid_loader = build_loader("valid")
    test_loader = build_loader("test")

    return train_loader, valid_loader, test_loader