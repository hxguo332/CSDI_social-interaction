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
    def __init__(self, *args, augmentation_mode='crop_and_resize', resize_to=(128, 128), **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_mode = augmentation_mode
        self.resize_to = resize_to
        if not self.load_scenario_map:
            raise ValueError("AugmentedSimulationDataset requires load_scenario_map=True")

    def __getitem__(self, index):
        index, scenario = self.get_index(index)
        track = self.data[scenario][index]

        observed_values, observed_masks, gt_masks, time_points, person_ids = self._parse_single_data(
            track, scenario, missing_ratio=self.missing_ratio, missing_strategy=self.missing_strategy)

        scen_map_raw = self.scen_map[scenario]
        desired_crop_size = self.resize_to

        # Use maybe_augment_sample to get the processed scenario map and adjusted track data
        scen_map_processed, adjusted_values = self.maybe_augment_sample(
            scen_map_raw, observed_values, desired_crop_size
        )

        s = {
            'observed_data': adjusted_values,
            'observed_mask': observed_masks,
            'gt_mask': gt_masks,
            'timepoints': np.arange(self.data_length),
            'person_ids': person_ids,
            'scen_map': scen_map_processed,
            'scen_map_scale': np.full(2, self.scen_map_base_scale),
            'resize_shape': self.resize_to
        }
        return s
    def random_crop_and_resize_with_track(self, scen_map_raw, observed_values, desired_crop_size):
        """
        Crop the scenario map to cover the track (from observed_values) with a random margin,
        pad if needed, and resize to desired_crop_size, keeping aspect ratio.
        Adjust the track coordinates accordingly.
        """
        import cv2
        import numpy as np
        H, W, _ = scen_map_raw.shape
        min_x, min_y = observed_values.min(axis=0)
        max_x, max_y = observed_values.max(axis=0)

        # Calculate bounding box covering the track with random margins
        margin_x = random.randint(0, int(0.1 * W))
        margin_y = random.randint(0, int(0.1 * H))

        left = max(int(min_x) - margin_x, 0)
        right = min(int(max_x) + margin_x, W)
        top = max(int(min_y) - margin_y, 0)
        bottom = min(int(max_y) + margin_y, H)

        crop_w = right - left
        crop_h = bottom - top

        # Ensure minimum size matching desired_crop_size
        out_h, out_w = desired_crop_size
        pad_right = max(out_w - crop_w, 0)
        pad_bottom = max(out_h - crop_h, 0)

        cropped_map = scen_map_raw[top:bottom, left:right]
        if pad_right > 0 or pad_bottom > 0:
            cropped_map = cv2.copyMakeBorder(
                cropped_map,
                0, pad_bottom,
                0, pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )

        # Resize while keeping aspect ratio
        scale_x = out_w / cropped_map.shape[1]
        scale_y = out_h / cropped_map.shape[0]
        scale = min(scale_x, scale_y)
        new_size = (int(cropped_map.shape[1] * scale), int(cropped_map.shape[0] * scale))
        resized_map = cv2.resize(cropped_map, (new_size[0], new_size[1]), interpolation=cv2.INTER_LINEAR)

        # Pad to desired size if needed
        final_map = np.zeros((out_h, out_w, 3), dtype=resized_map.dtype)
        final_map[:new_size[1], :new_size[0]] = resized_map

        # Adjust track coordinates
        adjusted_values = observed_values - np.array([left, top])
        # Also scale the adjusted values accordingly
        adjusted_values = adjusted_values * scale

        return final_map, adjusted_values

    def maybe_augment_sample(self, scen_map_raw, observed_values, desired_crop_size):
        """
        With 50% probability, apply random_crop_and_resize_with_track, otherwise return original data.
        """
        if random.random() < 0.5:
            # Use original map and coordinates
            return scen_map_raw, observed_values
        else:
            return self.random_crop_and_resize_with_track(scen_map_raw, observed_values, desired_crop_size)