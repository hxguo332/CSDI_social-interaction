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

    def random_crop_and_resize_with_track(self, scen_map_raw, observed_values, observed_masks, desired_crop_size):
        """
        Crop the scenario map to cover the track (from observed_values) with an asymmetric random margin,
        try to match the desired aspect ratio as closely as possible, pad if needed, and resize to desired_crop_size.
        Adjust the track coordinates accordingly.
        """
        H, W, _ = scen_map_raw.shape

        masked_value = np.where(observed_masks, observed_values, np.nan)

        # Bounding box around the track
        min_x, min_y = np.nanmin(masked_value, axis=0) * self.scen_map_base_scale
        max_x, max_y = np.nanmax(masked_value, axis=0) * self.scen_map_base_scale
        min_y, max_y = H - max_y, H - min_y  # Flip y-coordinate to match image coordinates, the code need to be in the same line so that the min_y and max_y are flipped correctly

        # Ensure min_x, max_x, min_y, max_y are integers
        min_x = np.floor(min_x)
        max_x = np.ceil(max_x)
        min_y = np.floor(min_y)
        max_y = np.ceil(max_y)

        bbox_w = max_x - min_x
        bbox_h = max_y - min_y

        # If the bounding box is too small, expand it to a minimum size
        min_size = 30  # Minimum size for the bounding box
        if bbox_w < min_size:
            print(f"Original min_x: {min_x}, max_x: {max_x}, bbox_w: {bbox_w}")
            min_x -= max(0, (min_size - bbox_w) / 2)
            max_x += min(W, (min_size - bbox_w) / 2)
            bbox_w = max_x - min_x
            print(f"Adjusted bbox width: {bbox_w} (min_x: {min_x}, max_x: {max_x})")
        if bbox_h < min_size:
            print(f"Original min_y: {min_y}, max_y: {max_y}, bbox_h: {bbox_h}")
            min_y -= max(0, (min_size - bbox_h) / 2)
            max_y += min(H, (min_size - bbox_h) / 2)
            bbox_h = max_y - min_y
            print(f"Adjusted bbox height: {bbox_h} (min_y: {min_y}, max_y: {max_y})")


        scale_param = 1
        # Apply asymmetric random margins
        margin_left = random.randint(0, min(int(scale_param * bbox_w), int(min_x)))
        margin_right = random.randint(0, min(int(scale_param * bbox_w), int(W - max_x)))
        margin_top = random.randint(0, min(int(scale_param * bbox_h), int(min_y)))
        margin_bottom = random.randint(0, min(int(scale_param * bbox_h), int(H - max_y)))

        # calculate the margin_right and margin_bottom to achieve the desired aspect ratio
        target_aspect = desired_crop_size[1] / desired_crop_size[0]
        current_aspect = (max_x - min_x + margin_left) / (max_y - min_y + margin_top)
        if current_aspect > target_aspect:
            # Too wide: adjust margin_bottom to match target aspect ratio
            margin_bottom = int((max_x - min_x + margin_left + margin_right) / target_aspect - (max_y - min_y + margin_top))
            margin_bottom = max(margin_bottom, 0)  # Ensure non-negative
        else:
            # Too tall: adjust margin_right to match target aspect ratio
            margin_right = int((max_y - min_y + margin_top + margin_bottom) * target_aspect - (max_x - min_x + margin_left))
            margin_right = max(margin_right, 0) # Ensure non-negative

        # Proposed crop box
        left = max(int(min_x) - margin_left, 0)
        right = min(int(max_x) + margin_right, W)
        top = max(int(min_y) - margin_top, 0)
        bottom = min(int(max_y) + margin_bottom, H)

        crop_w = right - left
        crop_h = bottom - top

        # Compute padding needed to achieve target aspect ratio
        current_aspect = crop_w / crop_h if crop_h != 0 else 1.0
        target_aspect = desired_crop_size[1] / desired_crop_size[0]

        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        if current_aspect > target_aspect:
            # Too wide: pad height
            target_crop_h = int(crop_w / target_aspect)
            pad_total = max(target_crop_h - crop_h, 0)
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
        elif current_aspect < target_aspect:
            # Too tall: pad width
            target_crop_w = int(crop_h * target_aspect)
            pad_total = max(target_crop_w - crop_w, 0)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left

        # Crop and pad
        cropped_map = scen_map_raw[top:bottom, left:right]
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            padded_map = cv2.copyMakeBorder(
                cropped_map,
                pad_top, pad_bottom,
                pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )
        else:
            padded_map = cropped_map
        print(f"Crop: ({left}, {top}) to ({right}, {bottom}), Padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
        print("padded map shape:", padded_map.shape)
        if padded_map.shape[0] == 0 or padded_map.shape[1] == 0:
            print(f"Warning: Padded map has zero height or width. Crop: ({left}, {top}) to ({right}, {bottom}), Padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
            print(f"track range: x=({observed_values[:, 0].min()}, {observed_values[:, 0].max()}), y=({observed_values[:, 1].min()}, {observed_values[:, 1].max()})")
        print("padded aspect ratio:", padded_map.shape[1] / padded_map.shape[0], "target aspect ratio:", desired_crop_size[1] / desired_crop_size[0])

        # Resize directly to target shape
        final_map = cv2.resize(padded_map, desired_crop_size[::-1], interpolation=cv2.INTER_LINEAR)

        # Adjust track coordinates
        adjusted_values = observed_values * self.scen_map_base_scale 
        adjusted_values[:, 1] = H - adjusted_values[:, 1]  # Flip y-coordinate to match image coordinates
        adjusted_values = adjusted_values - np.array([left, top])  # Adjust for crop
        adjusted_values[:, 0] += pad_left
        adjusted_values[:, 1] += pad_top

        # Compute rescale factor from padded crop to output
        scale_x = desired_crop_size[1] / padded_map.shape[1]
        scale_y = desired_crop_size[0] / padded_map.shape[0]
        adjusted_values[:, 0] *= scale_x
        adjusted_values[:, 1] *= scale_y
        print(f"Final map shape: {final_map.shape}, Crop size: {desired_crop_size}")
        print(f"Scale: x={scale_x}, y={scale_y}")

        return final_map, adjusted_values, (scale_x + scale_y) / 2  # average scale

    def convert_to_track_coordinates(self, observed_values, H):
        """
        Convert observed values to track coordinates by flipping the y-coordinate.
        """
        adjusted_values = observed_values.copy() * self.scen_map_base_scale
        adjusted_values[:, 1] = H - adjusted_values[:, 1]
        return adjusted_values

    def maybe_augment_sample(self, scen_map_raw, observed_values, observed_masks, desired_crop_size):
        """
        With 50% probability, apply nandom_crop_and_resize_with_track, otherwise return original data.
        """
        if random.random() < 0.5:
            # Use original map and coordinates
            observed_values = self.convert_to_track_coordinates(observed_values, scen_map_raw.shape[0])
            # No augmentation applied, return original map and values
            return scen_map_raw, observed_values, self.scen_map_base_scale # 1.0 scale means no scaling applied
        else:
            final_map, adjusted_values, scale = self.random_crop_and_resize_with_track(scen_map_raw, observed_values, observed_masks, desired_crop_size)
            return final_map, adjusted_values, scale