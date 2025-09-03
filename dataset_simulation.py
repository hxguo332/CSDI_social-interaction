import os
from torch.utils.data import DataLoader, Dataset, Sampler
import pandas as pd
import numpy as np
import json
from scenario_map import create_scenario_map_3channel


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
            zero_based_position (bool): If True, positions will be adjusted to be zero-based. The scenario map is adjusted with zero-based positions by default. So set this to True if you need to use the scenario map.
            scen_map_scale (int): The base scale to create the scenario map.
        """
        self.data_length = data_length
        self.scenarios = scenarios if scenarios else ["1-1","2-1","2-2","2-3","3-1","3-2", "4-1"]
        self.missing_strategy = missing_strategy
        self.missing_ratio = missing_ratio
        self.zero_based_position = zero_based_position
        self.origin_position = {k: None for k in self.scenarios}
        self.data_folder = data_folder or "./data/simulation_data/"
        self.subset_split_seed = subset_split_seed
        self.mode = mode
        self.load_scenario_map = load_scenario_map
        self.data = {}
        self.scen_map = {}
        self.scen_map_info = {}
        self.scen_map_base_scale = scen_map_scale # The base scale to create the scenario map

        for scenario in self.scenarios:
            person_info, sim_info = self._load_data(scenario)
            sim_info = self._filter_data(person_info, sim_info)
            self.data[scenario] = self._split_data(sim_info, data_length)
            self.data[scenario] = self.split_data_in_subsets(self.data[scenario])

            self.scen_map_info[scenario] = self._load_scen_map(scenario)
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
            random: randomly set some percentage as missing targets
            middle: set the middle of the sequence as missing targets
            all_but_two_end: set all but the two ends of the sequence as missing targets
            end: set the end of the sequence as missing targets

        missing_ratio (float): the ratio of missing values to use
        Output:
            observed_values: np.array, shape: (T x N)
            observed_masks: np.array, shape: (T x N)
            gt_masks: np.array, shape: (T x N)
            time_points: np.array, shape: (T)
            person_ids: np.array, shape: (1)
        
        """
        observed_values = np.array(data[['posX','posY']])

        # Convert the position to zero-based if needed
        if self.zero_based_position:
            if self.origin_position[scenario] is None:
                self.origin_position[scenario] = np.array([min(self.scen_map_info[scenario]['position']['p1'][0], self.scen_map_info[scenario]['position']['p2'][0]), min(self.scen_map_info[scenario]['position']['p1'][1], self.scen_map_info[scenario]['position']['p2'][1])])
            observed_values = observed_values - self.origin_position[scenario]

        observed_masks = ~np.isnan(observed_values)
        observed_values = np.nan_to_num(observed_values)

        # set some percentage as missing targets
        masks = observed_masks.reshape(-1).copy()
        obs_indices = np.where(masks)[0].tolist()
        if missing_strategy == "random":
            missing_indices =np.random.choice(
                obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
            )
        elif missing_strategy == "middle":
            missing_indices = obs_indices[len(obs_indices)*(1-missing_ratio)//2:len(obs_indices)*(1+missing_ratio)//2]
        elif missing_strategy == "all_but_two_end":
            missing_indices = obs_indices[2:-2] # there are x,y in the two end
        elif missing_strategy == "end":
            missing_indices = obs_indices[len(obs_indices)*(1-missing_ratio)//2*2:] # keep the index to a even number
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
        #person_ids = np.array(data['personID'])
        #person_ids = np.nan_to_num(person_ids)
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
            # for unit test, we use the first 50 samples
            return data[:20]
        else:
            raise ValueError(f"Mode {self.mode} is not available. Choose from ['train', 'valid', 'test']")

    def _split_data(self, sim_info, desired_length):
        """
        Split the data to the desired length, the split should keep the end of each track. Pad nan to data if the length is less than the desired length.
        Args:
            sim_info (pd.DataFrame): have the following columns: time,personID,posX,posY
            desired_length (int): the desired length of the split data
        Output:
            splited_data (list of pd.DataFrame): each DataFrame has the same length,
            and the length is equal to desired_length. If the length of the track is less than
            desired_length, pad the data with NaN.
            Each DataFrame has the following columns: time,personID,posX,posY
        """
        # Get the unique personIDs
        person_ids = sim_info['personID'].unique()
        # Initialize the splited data
        splited_data = []

        for person_id in person_ids:
            # Get the data for the current person_id
            person_data = sim_info[sim_info['personID'] == person_id]
            # Get the length of the data
            data_length = len(person_data)
            # Calculate the number of splits
            num_splits = data_length // desired_length
            # Calculate the remainder
            remainder = data_length % desired_length
            # Split the data
            for i in range(num_splits):
                splited_data.append(person_data.iloc[i*desired_length:(i+1)*desired_length])
            # Append and pad the remainder
            if remainder > 0:
                remaining_data = person_data.iloc[-remainder:]
                padding = desired_length - remainder
                padding_data = pd.DataFrame(np.nan, index=np.arange(padding), columns=remaining_data.columns)
                data_with_padding = pd.concat([remaining_data, padding_data])
                splited_data.append(data_with_padding)


        return splited_data


    def _filter_data(self, person_info, sim_info, time_limit=200):
        """
        Example of the input data:
            person_info (pd.DataFrame):
            PersonID	PersonType	Weight	Radius	MaxSpeed	GenTime	GenID	numGoal	GoalIDarray
            1	3	60.0000	0.210000	0.00000	0.10	3	1	3
            2	3	60.0000	0.210000	0.00000	0.10	4	1	4

            sim_info (pd.DataFrame):
            time	personID	posX	posY
            0.1	    1	        34.0	14.0
            0.1	    2	        61.0	11.0
            0.2	    1	        34.0	14.0
        """
        # remove the person who is standing (PersonType=3) in sim_info
        # Get the list of PersonIDs where PersonType == 3
        person_ids_to_remove = person_info[person_info['PersonType'] == 3]['PersonID']

        # Filter out rows in sim_info where personID matches those in person_ids_to_remove
        filtered_sim_info = sim_info[~sim_info['personID'].isin(person_ids_to_remove)]

        # Keep the sim_info rows where time is less than or equal to time_limit
        filtered_sim_info = filtered_sim_info[filtered_sim_info['time'] <= time_limit]
        return filtered_sim_info
            
    def _load_data(self, scenario):
        assert scenario in self.scenarios, f"Scenario {scenario} is not available. Please choose from {self.scenarios}"
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
        rescale = False
        index, scenario = self.get_index(index)
        track = self.data[scenario][index]

        # Parse the track data to obtain:
        # - observed_values: Nx2 numpy array of (x, y) positions
        # - observed_masks: Nx2 boolean array indicating if data is observed (1) or missing (0)
        # - gt_masks: Nx2 boolean array indicating the training target (1 for visible, 0 for prediction)
        # - time_points: Array of time points corresponding to each observation
        # - person_ids: Array containing IDs of the persons associated with the track
        observed_values, observed_masks, gt_masks, time_points, person_ids = self._parse_single_data(track, scenario, missing_ratio=self.missing_ratio, missing_strategy=self.missing_strategy)

        
        # Store all processed information in a dictionary
        s = {
            'observed_data': observed_values, # this is the left-down coordinates
            'observed_mask': observed_masks,
            'gt_mask': gt_masks,
            #'timepoints': time_points, # should I use the actual timepoints or the index? 
            'timepoints': np.arange(self.data_length), # should I use the actual timepoints or the index? 
            'person_ids': person_ids,
        }
        if self.load_scenario_map:
            s['scen_map'] = self.scen_map[scenario]
            if not rescale:
                # if there is no rescale for the scenario map, we use the original scale
                s['scen_map_scale'] = np.full(2, self.scen_map_base_scale) # should be 2D
            else:
                # if there is a rescale for the scenario map, we use the rescaled scale
                s['scen_map_scale'] = rescale
            H = s['scen_map'].shape[0]
            # convert observed values to left-top coordinates from the left-down coordinates
            observed_values =self.convert_to_track_coordinates(observed_values, H)
            s['observed_data'] = observed_values 
        return s

    def get_index(self, index):
        # given an index, return the scenario and the index in the scenario
        for scenario in self.scenarios:
            if index < len(self.data[scenario]):
                return index, scenario
            index -= len(self.data[scenario])
        #raise ValueError(f"Index {index} is out of range.")

    def __len__(self):
        return sum([len(self.data[sc]) for sc in self.scenarios])

    def convert_to_track_coordinates(self, observed_values, H):
        """
        Convert observed values to track coordinates by flipping the y-coordinate.
        """
        adjusted_values = observed_values.copy() * self.scen_map_base_scale
        adjusted_values[:, 1] = H - adjusted_values[:, 1]
        return adjusted_values

def get_dataloader(data_length, seed, scenarios=None,batch_size=8, load_scenario_map=False, zero_based_position=False):
    dataset = Simulation_Dataset(data_length, subset_split_seed=seed, scenarios=scenarios, mode='train', load_scenario_map=load_scenario_map, zero_based_position=zero_based_position)
    train_loader = ScenarioBatchDataLoader(
        dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Simulation_Dataset(data_length,subset_split_seed=seed, scenarios=scenarios, mode='valid', load_scenario_map=load_scenario_map, zero_based_position=zero_based_position)
    valid_loader = ScenarioBatchDataLoader(
        valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Simulation_Dataset(data_length,subset_split_seed=seed, scenarios=scenarios, mode='test', load_scenario_map=load_scenario_map, zero_based_position=zero_based_position)
    test_loader = ScenarioBatchDataLoader(
        test_dataset, batch_size=batch_size, shuffle=0)

    #scaler = torch.from_numpy(dataset.std_data).to(device).float()
    #mean_scaler = torch.from_numpy(dataset.mean_data).to(device).float()

    return train_loader, valid_loader, test_loader

def get_unit_test_dataloader(data_length, seed, scenarios=None,batch_size=8, load_scenario_map=False, zero_based_position=False):
    dataset = Simulation_Dataset(data_length, subset_split_seed=seed, scenarios=scenarios, mode='unit_test', load_scenario_map=load_scenario_map, zero_based_position=zero_based_position)
    dataloader = ScenarioBatchDataLoader(
        dataset, batch_size=batch_size, shuffle=1)
    return dataloader

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

        self.scenario_order = []  # Store the scenario order for the epoch
        self._prepare_epoch()  # Prepare indices when sampler is initialized

    def _prepare_epoch(self):
        self.scenario_order = []

        # For each scenario, generate all batches with padding if needed
        for scenario, indices in self.indices_by_scenario.items():
            if self.shuffle:
                np.random.shuffle(indices)

            # Padding if not divisible by batch_size
            if len(indices) % self.batch_size != 0:
                padding_size = self.batch_size - (len(indices) % self.batch_size)
                indices += np.random.choice(indices, padding_size).tolist()

            scenario_batches = [
                indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)
            ]
            self.scenario_order += [(scenario, batch) for batch in scenario_batches]

        # Shuffle the order of scenarios for each epoch
        if self.shuffle:
            np.random.shuffle(self.scenario_order)

    def __iter__(self):
        self._prepare_epoch()  # Prepare indices for the new epoch
        for scenario, batch_indices in self.scenario_order:
            for idx in batch_indices:
                yield idx

    def __len__(self):
        return sum([len(batch_indices) for _, batch_indices in self.scenario_order])