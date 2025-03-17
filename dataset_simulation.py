import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import json
from scenario_map import create_scenario_map_3channel


class Simulation_Dataset(Dataset):
    def __init__(self, data_length, scenario="1-1", data_folder=None, subset_split_seed=1, mode="train", missing_strategy="all_but_two_end", missing_ratio=0.1, load_scenario_map=False, zero_based_position=False):
        """
        Args:
            data_length (int): the length of the data sequence
            scenario (str): the scenario to load data from
            data_folder (str): the folder containing the data
            subset_split_seed (int): the seed to split the data into train, valid and test
            mode (str): the mode to load the data from ['train','valid','test']
            missing_strategy (str): the missing strategy to create gt_masks
            missing_ratio (float): the ratio of missing values to use
            load_scenario_map (bool): whether to load the scenario map
            zero_based_position (bool): whether the position is converted to zero-based according to the scenario map
        """
        self.data_length = data_length
        self.scenarios = ["1-1","2-1","2-2","2-3","3-1","3-2", "4-1"]
        self.missing_strategy = missing_strategy
        self.missing_ratio = missing_ratio
        self.zero_based_position = zero_based_position
        self.origin_position = None
        self.scenario = scenario
        if data_folder is None:
            self.data_folder = "./data/simulation_data/"
        else:
            self.data_folder = data_folder

        person_info, sim_info = self._load_data(scenario)

        # Remove the standing person and keep the data within the time limit
        sim_info = self._filter_data(person_info, sim_info)

        # Split the data
        self.data = self._split_data(sim_info, data_length)
        self.subset_split_seed = subset_split_seed
        self.mode = mode
        self.data = self.split_data_in_subsets()

        self.load_scenario_map = load_scenario_map
        self.scen_map_info = self._load_scen_map(scenario)
        if self.load_scenario_map:
            self.scen_map = {scenario: create_scenario_map_3channel(self.scen_map_info, scale=10)}

    def _load_scen_map(self, scenario):
        assert scenario in self.scenarios, f"Scenario {scenario} is not available. Please choose from {self.scenarios}"
        path = os.path.join(self.data_folder, f"Scenario{scenario}/scenario_map.json")
        with open(path, "r") as f:
            scenario_map = json.load(f)
        return scenario_map


    def split_data_in_subsets(self):
        np.random.seed(self.subset_split_seed)
        np.random.shuffle(self.data)
        if self.mode == "train":
            self.data = self.data[:int(0.7*len(self.data))]
        elif self.mode == "valid":
            self.data = self.data[int(0.7*len(self.data)):int(0.85*len(self.data))]
        elif self.mode == "test":
            self.data = self.data[int(0.85*len(self.data)):]
        else:
            raise ValueError(f"Mode {self.mode} is not available. Please choose from ['train','valid','test']")
        return self.data


    def _parse_all_data(self, data):
        """
        Parse the data to the desired format.
        data (pd.DataFrame): have the following columns: time,personID,posX,posY
        Output:
            observed_values: np.array, shape: (T x N)
            observed_masks: np.array, shape: (T x N)
            gt_masks: np.array, shape: (T x N)
            time_points: np.array, shape: (T)
            person_ids: np.array, shape: (1)
        
        """
        for i in range(len(self.data)):
            if i == 0:
                observed_values = np.array(data[i][['posX','posY']])
                observed_masks = ~np.isnan(observed_values)
                gt_masks = observed_masks.copy()
                time_points = np.array(data[i]['time'])
                person_ids = np.array(data[i]['personID'])
            else:
                observed_values = np.vstack((observed_values, np.array(data[i][['posX','posY']])))
                observed_masks = np.vstack((observed_masks, np.array(data[i][['posX','posY']])))
                gt_masks = np.vstack((gt_masks, np.array(data[i][['posX','posY']])))
                time_points = np.hstack((time_points, np.array(data[i]['time'])))
                person_ids = np.hstack((person_ids, np.array(data[i]['personID'])))
        return observed_values, observed_masks, gt_masks, time_points, person_ids

    def _parse_single_data(self, data, missing_strategy="random", missing_ratio=0.1):
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
            if self.origin_position is None:
                self.origin_position = np.array([min(self.scen_map_info['position']['p1'][0], self.scen_map_info['position']['p2'][0]), min(self.scen_map_info['position']['p1'][1], self.scen_map_info['position']['p2'][1])])
            observed_values = observed_values - self.origin_position

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
        person_ids = np.array(data['personID'])
        person_ids = np.nan_to_num(person_ids)
        return observed_values, observed_masks, gt_masks, time_points, person_ids

    def _split_data(self, sim_info, desired_length):
        """
        Split the data to the desired length, the split should keep the end of each track. Pad nan to data if the length is less than the desired length.
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
            path_info = os.path.join(self.data_folder, "data/simulation_data/Scenario4-1/outputdata-1.6.2021/outputPersonInfo_clean.csv")
        else:
            path_sim = os.path.join(self.data_folder, f"Scenario{scenario}/output data-2.15(yamada@vri)/simulationLog_clean.csv")
            path_info = os.path.join(self.data_folder,f"Scenario{scenario}/output data-2.15(yamada@vri)/outputPersonInfo_clean.csv")
        person_info = pd.read_csv(path_info)
        sim_info = pd.read_csv(path_sim)
        return person_info, sim_info

    def _data_preprocessing(self, data):
        """ Preprocess the data by normalizing the position"""
        pass
        
    def __getitem__(self, index):
        track = self.data[index]
        observed_values, observed_masks, gt_masks, time_points, person_ids = self._parse_single_data(track, missing_ratio=self.missing_ratio, missing_strategy=self.missing_strategy)
        s = {
            'observed_data': observed_values,
            'observed_mask': observed_masks,
            'gt_mask': gt_masks,
            #'timepoints': time_points, # should I use the actual timepoints or the index? 
            'timepoints': np.arange(self.data_length), # should I use the actual timepoints or the index? 
            'person_ids': person_ids,
        }
        if self.load_scenario_map:
            s['scen_map'] = self.scen_map[self.scenario]
        return s
    def __len__(self):
        return len(self.data)

def get_dataloader(data_length, seed, scenario="1-1",batch_size=8, load_scenario_map=False, zero_based_position=False):
    dataset = Simulation_Dataset(data_length, subset_split_seed=seed, scenario=scenario, mode='train', load_scenario_map=load_scenario_map, zero_based_position=zero_based_position)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Simulation_Dataset(data_length,subset_split_seed=seed, scenario=scenario, mode='valid', load_scenario_map=load_scenario_map, zero_based_position=zero_based_position)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Simulation_Dataset(data_length,subset_split_seed=seed, scenario=scenario, mode='test', load_scenario_map=load_scenario_map, zero_based_position=zero_based_position)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=0)

    #scaler = torch.from_numpy(dataset.std_data).to(device).float()
    #mean_scaler = torch.from_numpy(dataset.mean_data).to(device).float()

    return train_loader, valid_loader, test_loader