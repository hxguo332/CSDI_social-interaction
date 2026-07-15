import os
from torch.utils.data import DataLoader, Dataset, Sampler
import pandas as pd
import numpy as np
import json
# Assuming these helper scripts are available
from scenario_map import create_scenario_map_3channel
from sdf import generate_sdf


class Simulation_Dataset(Dataset):
    def __init__(self, data_length, scenarios=None, data_folder=None, subset_split_seed=1, mode="train", missing_strategy="all_but_two_end", missing_ratio=0.1, load_scenario_map=False, zero_based_position=True, scen_map_scale=10):
        """
        Simulation_Dataset is a dataset class for loading and processing simulation data.
        Args:
            data_length (int): The desired length of the data for each track.
            scenarios (list): List of scenarios to load. If None, all scenarios are loaded.
            data_folder (str): The folder where the simulation data is stored. If None, it defaults to "./data/simulation_data/".
            subset_split_seed (int): The seed for splitting the dataset into subsets (train, valid, test).
            mode (str): The mode of the dataset. Can be "train", "valid", "test", or "unit_test".
            missing_strategy (str): The strategy to create ground truth masks for missing data.
                Options: "random", "middle", "all_but_two_end", "end".
            missing_ratio (float): The ratio of missing values to use when applying the missing strategy.
            load_scenario_map (bool): Whether to load the scenario map for each scenario.
            zero_based_position (bool): If True, positions will be adjusted to be zero-based based on scen_map_info boundaries.
                Set this to True if you need to use the scenario map.
            scen_map_scale (int): The base scale to create the scenario map (World-to-Pixel factor).
        """
        self.data_length = data_length
        self.scenarios = scenarios if scenarios else ["1-1","2-1","2-2","2-3","3-1","3-2", "4-1"]
        self.missing_strategy = missing_strategy
        self.missing_ratio = missing_ratio
        self.zero_based_position = zero_based_position
        
        # FIX: Define unified, static origin strictly based on map metadata, not min posY
        self._unified_origins = {} 
        self.data_folder = data_folder or "./data/simulation_data/"
        self.subset_split_seed = subset_split_seed
        self.mode = mode
        self.load_scenario_map = load_scenario_map
        self.data = {}
        self.sim_info_by_scenario = {}
        self.scen_map = {}
        self.scen_map_info = {}
        self.scen_map_base_scale = scen_map_scale # The factor to create the scenario map (w2p)

        for scenario in self.scenarios:
            person_info, sim_info = self._load_data(scenario)
            sim_info = self._filter_data(person_info, sim_info)
            self.sim_info_by_scenario[scenario] = sim_info.copy()
            self.data[scenario] = self._split_data(sim_info, data_length)
            self.data[scenario] = self.split_data_in_subsets(self.data[scenario])

            self.scen_map_info[scenario] = self._load_scen_map(scenario)
            
            # Strict Origin definition: Map World boundaries metadata
            pos = self.scen_map_info[scenario]["position"]
            self._unified_origins[scenario] = np.array([
                min(pos["p1"][0], pos["p2"][0]),
                min(pos["p1"][1], pos["p2"][1]),
            ])

            if self.load_scenario_map:
                self.scen_map[scenario] = create_scenario_map_3channel(self._load_scen_map(scenario), scale=self.scen_map_base_scale)

    def _load_scen_map(self, scenario):
        assert scenario in self.scenarios, f"Scenario {scenario} is not available."
        path = os.path.join(self.data_folder, f"Scenario{scenario}/scenario_map.json")
        with open(path, "r") as f:
            return json.load(f)

    def __len__(self):
        return sum([len(self.data[sc]) for sc in self.scenarios])

    def _parse_single_data(self, data, scenario, missing_strategy="random", missing_ratio=0.1):
        """
        Parse the data to the desired format.
        data (pd.DataFrame): have the following columns: time,personID,posX,posY
        missing_strategy (str): the missing strategy to create gt_masks
        Output:
            observed_values: np.array, shape: (T x N) in ABSOLUTE World Coordinates
            observed_masks: np.array, shape: (T x N)
            gt_masks: np.array, shape: (T x N)
            time_points: np.array, shape: (T)
            person_ids: np.array, shape: (1)
        """
        # Maintain original absolute world coordinates (posX, posY)
        observed_values = np.array(data[['posX','posY']])

        # REMOVED zero_based_position logic from here. 
        # All Agent types (Ego, Neighbors) must maintain unified base absolute world coordinates
        # until unified normalization logic in getitem/get_neighbors.

        observed_masks = ~np.isnan(observed_values)
        observed_values = np.nan_to_num(observed_values)

        masks = observed_masks.reshape(-1).copy()
        obs_indices = np.where(masks)[0].tolist()
        if missing_strategy == "random":
            valid_steps = np.where(observed_masks[:, 0] & observed_masks[:, 1])[0]
            num_missing_steps = int(len(valid_steps) * missing_ratio)
            if num_missing_steps > 0:
                missing_steps = np.random.choice(valid_steps, num_missing_steps, replace=False)
                missing_indices = []
                for t in missing_steps:
                    missing_indices.extend([2 * t, 2 * t + 1])
            else:
                missing_indices = []
        elif missing_strategy == "middle":
            start = int(len(obs_indices) * (1 - missing_ratio) // 2)
            end = int(len(obs_indices) * (1 + missing_ratio) // 2)
            missing_indices = obs_indices[start:end]
        elif missing_strategy == "all_but_two_end":
            missing_indices = obs_indices[2:-2] # there are x,y in the two end
        elif missing_strategy == "end":
            start = int(len(obs_indices) * (1 - missing_ratio) // 2) * 2
            missing_indices = obs_indices[start:] # keep the index to an even number
        else:
            raise ValueError(f"Missing strategy {missing_strategy} is not available. Please choose from ['random','middle','all_but_two_end','end']")
        masks[missing_indices] = False
        gt_masks = masks.reshape(observed_masks.shape)

        time_points = np.array(data['time'])
        time_points = np.nan_to_num(time_points)
        person_ids = data['personID'].to_numpy()
        if np.issubdtype(person_ids.dtype, np.floating) and np.isnan(person_ids).any():
            person_ids = np.where(np.isnan(person_ids), -1, person_ids)

        person_ids = person_ids.astype(np.int64, copy=False)
        return observed_values, observed_masks, gt_masks, time_points, person_ids

    def split_data_in_subsets(self, data):
        np.random.seed(self.subset_split_seed)
        np.random.shuffle(data)
        if self.mode == "train":
            return data[:int(0.7*len(data))]
        elif self.mode == "valid":
            return data[int(0.7*len(data)):int(0.85*len(data))]
        elif self.mode == "test":
            return data[int(0.85*len(data)):]
        elif self.mode == "unit_test":
            return data[:20]
        else:
            raise ValueError(f"Mode {self.mode} is not available. Choose from ['train', 'valid', 'test', 'unit_test']")

    def _split_data(self, sim_info, desired_length):
        person_ids = sim_info['personID'].unique()
        splited_data = []
        for person_id in person_ids:
            person_data = sim_info[sim_info['personID'] == person_id]
            data_length = len(person_data)
            num_splits = data_length // desired_length
            remainder = data_length % desired_length
            for i in range(num_splits):
                splited_data.append(person_data.iloc[i*desired_length:(i+1)*desired_length])
            if remainder > 0:
                remaining_data = person_data.iloc[-remainder:]
                padding = desired_length - remainder
                padding_data = pd.DataFrame(np.nan, index=np.arange(padding), columns=remaining_data.columns)
                data_with_padding = pd.concat([remaining_data, padding_data])
                splited_data.append(data_with_padding)
        return splited_data

    def _filter_data(self, person_info, sim_info, time_limit=200):
        person_ids_to_remove = person_info[person_info['PersonType'] == 3]['PersonID']
        filtered_sim_info = sim_info[~sim_info['personID'].isin(person_ids_to_remove)]
        filtered_sim_info = filtered_sim_info[filtered_sim_info['time'] <= time_limit]
        return filtered_sim_info
            
    def _load_data(self, scenario):
        assert scenario in self.scenarios, f"Scenario {scenario} is not available."
        if scenario == "4-1":
            path_sim = os.path.join(self.data_folder, "Scenario4-1/outputdata-1.6.2021/simulationLog_clean.csv")
            path_info = os.path.join(self.data_folder, "Scenario4-1/outputdata-1.6.2021/outputPersonInfo_clean.csv")
        else:
            path_sim = os.path.join(self.data_folder, f"Scenario{scenario}/output data-2.15(yamada@vri)/simulationLog_clean.csv")
            path_info = os.path.join(self.data_folder,f"Scenario{scenario}/output data-2.15(yamada@vri)/outputPersonInfo_clean.csv")
        person_info = pd.read_csv(path_info)
        sim_info = pd.read_csv(path_sim)
        return person_info, sim_info
        
    def __getitem__(self, index):
        index, scenario = self.get_index(index)
        track = self.data[scenario][index]
        # observed_values are now strictly Absolute World Coordinates
        observed_values, observed_masks, gt_masks, time_points, person_ids = self._parse_single_data(track, scenario, missing_ratio=self.missing_ratio, missing_strategy=self.missing_strategy)

        # Store scenario map pixel dimensions from scenario map info boundaries metadata
        pos = self.scen_map_info[scenario]["position"]
        width_world = abs(pos["p2"][0] - pos["p1"][0])
        height_world = abs(pos["p2"][1] - pos["p1"][1])
        W_px = int(width_world * self.scen_map_base_scale)
        H_px = int(height_world * self.scen_map_base_scale)

        # Unified Normalization Logic for ALL Agent types (Strict origin definition)
        observed_vals_norm, W_px_u, H_px_u = self._normalize_positions_with_visual_flip(
            observed_values, scenario, H_px, W_px
        )
        assert H_px == H_px_u and W_px == W_px_u

        s = {
            'observed_data': observed_vals_norm.astype(np.float32), # normalized visual coordinates [0,1], top=0
            'observed_mask': observed_masks,
            'gt_mask': gt_masks,
            'timepoints': np.arange(self.data_length),
            'person_ids': person_ids,
            'scen_map_scale': np.array([width_world, height_world], dtype=np.float32), # World Coordinate scale metadata
            'scenario': scenario # Ensure scenario string is returned
        }
        if self.load_scenario_map:
            s['scen_map'] = self.scen_map[scenario]
        return s

    def get_index(self, index):
        # given an index, return the scenario and the index in the scenario
        for scenario in self.scenarios:
            if index < len(self.data[scenario]):
                return index, scenario
            index -= len(self.data[scenario])

    def _normalize_positions_with_visual_flip(self, values, scenario, H_px, W_px):
        """Unified, Robust Normalization for both training (sdf) and visualization (alignment)."""
        # Strictly enforce defined unified origin for Absolute World Coordinates
        origin = self._unified_origins[scenario]
        vals_absolute_origin = values.copy() - origin # World Coordinate offset
        
        # Scale World coordinates to Map Pixels
        vals_px = vals_absolute_origin * self.scen_map_base_scale
        
        # Apply Y-axis insanity flip (screen sanity flip, 0 is top)
        # H_px - y maps world-min-Y (bottom) to screen-max-Y (bottom). 
        # But visually, 0 is top. So visually y_new = H_px - y.
        vals_px[:, 1] = H_px - vals_px[:, 1]
        
        # Normalize Map Pixels to [0,1]
        norm_div = np.array([W_px, H_px], dtype=np.float32)
        return (vals_px / np.clip(norm_div, 1e-8, None)).astype(np.float32), W_px, H_px

class ScenarioBatchDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, **kwargs):
        sampler = ScenarioBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, **kwargs)

class ScenarioBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.scenarios = dataset.scenarios
        self.indices_by_scenario = {}
        scenario_len = 0
        for scenario in self.scenarios:
            self.indices_by_scenario[scenario] = [
                i for i in range(scenario_len, scenario_len + len(dataset.data[scenario]))
            ]
            scenario_len += len(dataset.data[scenario])
        self.scenario_order = [] 
        self._prepare_epoch()

    def _prepare_epoch(self):
        self.scenario_order = []
        for scenario, indices in self.indices_by_scenario.items():
            if self.shuffle:
                np.random.shuffle(indices)
            if len(indices) % self.batch_size != 0:
                padding_size = self.batch_size - (len(indices) % self.batch_size)
                indices += np.random.choice(indices, padding_size).tolist()
            scenario_batches = [
                indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)
            ]
            self.scenario_order += [(scenario, batch) for batch in scenario_batches]
        if self.shuffle:
            np.random.shuffle(self.scenario_order)

    def __iter__(self):
        self._prepare_epoch()
        for scenario, batch_indices in self.scenario_order:
            for idx in batch_indices:
                yield idx

    def __len__(self):
        return sum([len(batch_indices) for _, batch_indices in self.scenario_order])


class SocialSimulation_Dataset(Simulation_Dataset):
    """Ego-centered social dataset ensuring visual sanity for all agents."""
    def __init__(
        self,
        data_length,
        scenarios=None,
        data_folder=None,
        subset_split_seed=1,
        mode="train",
        missing_strategy="all_but_two_end",
        missing_ratio=0.1,
        load_scenario_map=True,
        zero_based_position=True,
        scen_map_scale=10,
        max_neighbors=8,
        goal_sigma=0.035,
        gen_sdf=False,
    ):
        super().__init__(
            data_length=data_length,
            scenarios=scenarios,
            data_folder=data_folder,
            subset_split_seed=subset_split_seed,
            mode=mode,
            missing_strategy=missing_strategy,
            missing_ratio=missing_ratio,
            load_scenario_map=load_scenario_map,
            zero_based_position=zero_based_position,
            scen_map_scale=scen_map_scale,
        )
        self.max_neighbors = max_neighbors
        self.goal_sigma = goal_sigma
        self.gen_sdf = gen_sdf
        self.sdf_maps = {}
        if self.gen_sdf:
            if not self.load_scenario_map:
                raise ValueError("gen_sdf=True requires load_scenario_map=True")
            for scenario in self.scenarios:
                self.sdf_maps[scenario] = generate_sdf(self.scen_map[scenario])

    def __getitem__(self, index):
        # We reuse __getitem__ but add Social Branch processing
        index, scenario = self.get_index(index)
        track = self.data[scenario][index]
        observed_values, observed_masks, gt_masks, time_points, person_ids = self._parse_single_data(
            track, scenario, missing_ratio=self.missing_ratio, missing_strategy=self.missing_strategy
        )

        pos = self.scen_map_info[scenario]["position"]
        width_world = abs(pos["p2"][0] - pos["p1"][0])
        height_world = abs(pos["p2"][1] - pos["p1"][1])
        W_px = int(width_world * self.scen_map_base_scale)
        H_px = int(height_world * self.scen_map_base_scale)

        # ego pid
        ego_pid = int(person_ids[~np.isnan(person_ids)][0]) if len(person_ids) else -1
        
        # Social Branch requires Neighbors (N, L, 2) normalized to [0,1] visually sanity logic
        neighbor_data, neighbor_mask = self._get_neighbors(track, scenario, ego_pid, time_points, H_px, W_px)

        # Normalized Ego trajectory with visual sanity logic
        observed_vals_norm, W_px_u, H_px_u = self._normalize_positions_with_visual_flip(
            observed_values, scenario, H_px, W_px
        )
        assert H_px == H_px_u and W_px == W_px_u

        # Conflict features (requires internally correct distance calculation)
        conflict_features = self._compute_conflict_features(
            observed_vals_norm, observed_masks, neighbor_data, neighbor_mask
        )

        s = {
            "observed_data": observed_vals_norm.astype(np.float32), # Internal visual coordinates top=0
            "observed_mask": observed_masks,
            "gt_mask": gt_masks,
            "timepoints": np.arange(self.data_length),
            "person_ids": person_ids,
            "neighbor_data": neighbor_data, # Aligned internal visual coordinates top=0
            "neighbor_mask": neighbor_mask,
            "conflict_features": conflict_features,
            "goal_heatmap": self._make_goal_heatmap(observed_vals_norm, observed_masks, H_px, W_px),
            "scenario": scenario,
            'scen_map_scale': np.array([width_world, height_world], dtype=np.float32), # World Coordinate scale metadata
        }
        if self.load_scenario_map:
            s["scen_map"] = self.scen_map[scenario]
        if self.gen_sdf:
            s["sdf"] = self.sdf_maps[scenario]
        return s

    def _make_goal_heatmap(self, observed_values_norm, observed_masks, H, W):
        valid = observed_masks[:, 0] & observed_masks[:, 1]
        valid_idx = np.where(valid)[0]
        heat = np.zeros((H, W, 2), dtype=np.float32)
        if len(valid_idx) == 0:
            return heat
        # heat map uses visual coordinates (normalized imshow style)
        start = observed_values_norm[valid_idx[0]]
        end = observed_values_norm[valid_idx[-1]]
        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, H, dtype=np.float32),
            np.linspace(0.0, 1.0, W, dtype=np.float32),
            indexing="ij",
        )
        sigma2 = max(self.goal_sigma ** 2, 1e-8)
        # Heat maps use internally aligned coordinates
        heat[..., 0] = np.exp(-((xx - start[0]) ** 2 + (yy - start[1]) ** 2) / (2.0 * sigma2))
        heat[..., 1] = np.exp(-((xx - end[0]) ** 2 + (yy - end[1]) ** 2) / (2.0 * sigma2))
        return heat

    def _get_neighbors(self, ego_track, scenario, ego_pid, time_points, H_px, W_px):
        """Unified, Visual Sanity logic for Neighbors."""
        sim_info = self.sim_info_by_scenario[scenario]
        candidate = sim_info[
            (sim_info["time"].isin(time_points)) & (sim_info["personID"] != ego_pid)
        ]

        neighbor_rows = []
        for pid, grp in candidate.groupby("personID"):
            grp = grp.drop_duplicates("time").set_index("time")
            aligned = grp.reindex(time_points)
            values = aligned[["posX", "posY"]].to_numpy(dtype=np.float32) # Absolute World Coordinates
            masks = ~np.isnan(values).any(axis=1)
            if masks.sum() == 0:
                continue
            values = np.nan_to_num(values)
            
            # UNIFIED: Apply unified normalization and visual sanity logic to Neighbors
            norm_values, _, _ = self._normalize_positions_with_visual_flip(values, scenario, H_px, W_px)
            neighbor_rows.append((pid, norm_values, masks.astype(np.float32)))

        # ego_norm (needed for proximity scoring only, already normalized top=0)
        ego_values = ego_track[["posX", "posY"]].to_numpy(dtype=np.float32)
        ego_norm, _, _ = self._normalize_positions_with_visual_flip(np.nan_to_num(ego_values), scenario, H_px, W_px)

        scored = []
        for pid, values, mask in neighbor_rows:
            valid = mask > 0
            if valid.sum() == 0: continue
            # scored uses internal aligned distance (y_normalized top=0)
            dist = np.linalg.norm(ego_norm[valid] - values[valid], axis=-1)
            scored.append((float(dist.min()), pid, values, mask))
        scored.sort(key=lambda x: x[0])

        neighbor_data = np.zeros((self.max_neighbors, self.data_length, 2), dtype=np.float32)
        neighbor_mask = np.zeros((self.max_neighbors, self.data_length), dtype=np.float32)
        for i, (_, _, values, mask) in enumerate(scored[: self.max_neighbors]):
            neighbor_data[i] = values # normalized visual top=0 coordinates
            neighbor_mask[i] = mask
        return neighbor_data, neighbor_mask

    def _compute_conflict_features(self, ego_norm, ego_mask, neighbor_data, neighbor_mask):
        valid_ego = ego_mask[:, 0] & ego_mask[:, 1]
        if neighbor_mask.sum() == 0 or valid_ego.sum() < 2:
            return np.array([1.0, 1.0, 0.0], dtype=np.float32)

        # uses internally aligned [0,1] top=0 visual coordinates
        ego_vel = np.diff(ego_norm, axis=0, prepend=ego_norm[:1])
        min_dist, min_ttc = 1.0, 1.0
        heading_diffs = []
        for n in range(neighbor_data.shape[0]):
            valid = (neighbor_mask[n] > 0) & valid_ego
            if valid.sum() == 0: continue
            rel = neighbor_data[n] - ego_norm
            dist = np.linalg.norm(rel[valid], axis=-1)
            min_dist = min(min_dist, float(dist.min()))
            neigh_vel = np.diff(neighbor_data[n], axis=0, prepend=neighbor_data[n, :1])
            rel_vel = ego_vel - neigh_vel
            closing = -(rel * rel_vel).sum(axis=-1) / (np.linalg.norm(rel, axis=-1) + 1e-6)
            ttc = np.where(closing > 1e-6, np.linalg.norm(rel, axis=-1) / (closing + 1e-6), 1.0)
            min_ttc = min(min_ttc, float(np.clip(ttc[valid].min(), 0.0, 1.0)))
            ev = ego_vel[valid]; nv = neigh_vel[valid]
            cos = (ev * nv).sum(axis=-1) / (np.linalg.norm(ev, axis=-1) * np.linalg.norm(nv, axis=-1) + 1e-6)
            heading_diffs.append(np.arccos(np.clip(cos, -1.0, 1.0)) / np.pi)
        mean_heading = float(np.concatenate(heading_diffs).mean()) if heading_diffs else 0.0
        return np.array([min_dist, min_ttc, mean_heading], dtype=np.float32)


def get_social_dataloader(
    data_length, seed, scenarios=None, batch_size=8, 
    load_scenario_map=True, zero_based_position=True, # Design uses zero_based internal processing
    max_neighbors=8, gen_sdf=False, missing_strategy="end", missing_ratio=0.5,
):
    dataset = SocialSimulation_Dataset(
        data_length, subset_split_seed=seed, scenarios=scenarios, mode="train",
        load_scenario_map=load_scenario_map, zero_based_position=zero_based_position,
        max_neighbors=max_neighbors, gen_sdf=gen_sdf,
        missing_strategy=missing_strategy, missing_ratio=missing_ratio,
    )
    train_loader = ScenarioBatchDataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = SocialSimulation_Dataset(
        data_length, subset_split_seed=seed, scenarios=scenarios, mode="valid",
        load_scenario_map=load_scenario_map, zero_based_position=zero_based_position,
        max_neighbors=max_neighbors, gen_sdf=gen_sdf,
        missing_strategy=missing_strategy, missing_ratio=missing_ratio,
    )
    valid_loader = ScenarioBatchDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = SocialSimulation_Dataset(
        data_length, subset_split_seed=seed, scenarios=scenarios, mode="test",
        load_scenario_map=load_scenario_map, zero_based_position=zero_based_position,
        max_neighbors=max_neighbors, gen_sdf=gen_sdf,
        missing_strategy=missing_strategy, missing_ratio=missing_ratio,
    )
    test_loader = ScenarioBatchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

def get_dataloader(data_length, seed, scenarios=None, batch_size=8, load_scenario_map=False, zero_based_position=False, missing_strategy="all_but_two_end", missing_ratio=0.1):
    dataset = Simulation_Dataset(
        data_length, subset_split_seed=seed, scenarios=scenarios, 
        mode='train', load_scenario_map=load_scenario_map, 
        zero_based_position=zero_based_position,
        missing_strategy=missing_strategy, missing_ratio=missing_ratio
    )
    train_loader = ScenarioBatchDataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    valid_dataset = Simulation_Dataset(
        data_length, subset_split_seed=seed, scenarios=scenarios, 
        mode='valid', load_scenario_map=load_scenario_map, 
        zero_based_position=zero_based_position,
        missing_strategy=missing_strategy, missing_ratio=missing_ratio
    )
    valid_loader = ScenarioBatchDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = Simulation_Dataset(
        data_length, subset_split_seed=seed, scenarios=scenarios, 
        mode='test', load_scenario_map=load_scenario_map, 
        zero_based_position=zero_based_position,
        missing_strategy=missing_strategy, missing_ratio=missing_ratio
    )
    test_loader = ScenarioBatchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader