import json

def save_dict_to_json(data: dict, filename: str):
    """
    Saves a dictionary to a JSON file.
    
    :param data: Dictionary to save
    :param filename: Name of the JSON file to create
    """
    try:
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False)
        print(f"JSON file '{filename}' has been created successfully.")
    except Exception as e:
        print(f"Error writing JSON file: {e}")


obstacles_1_1 = [
    {"rectangle": {
        "p1": (28, 9),
        "p2": (30, 11),
    }},
    {"rectangle": {
        "p1": (54, 12),
        "p2": (58, 13),
    }},
    {"standing_person": (34, 14)},
    {"standing_person": (61, 10)},
    {"wall":[(0, 8),(84, 8), (84, 16), (0, 16), (0, 8)]},
    {"entrance":{ "p1": (0, 9), "p2": (0, 15)}},
    {"entrance":{ "p1": (84, 9), "p2": (84, 15)}},
]

if __name__ == "__main__":
    obstacles = [obstacles_1_1]
    path_dict = {"1-1": obstacles_1_1}
    for k, v in path_dict.items():
        json_filename = f"data/simulation_data/Scenario{k}/obstacles.json"
        # Save dictionary to JSON file
        save_dict_to_json(v, json_filename)