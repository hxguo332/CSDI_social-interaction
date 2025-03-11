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


scen_map_1_1 = {"obstacles":[
        {
            "type":"rectangle", 
            "p1": (28, 9),
            "p2": (30, 11),},
        {
            "type":"rectangle",
            "p1": (54, 12),
            "p2": (58, 13),
        },
    ],
    "forbidden_area":[
        {"type":"polygon", "points":[(0,7), (0,8),(84,8),(84,7)]},
        {"type":"rectangle", "p1":(0,16),"p2":(84, 17)},
    ],
    "standing_person": [(34, 14), (61, 10)], # (x, y)
    "wall":[{"segment":[(0, 8),(84, 8)]},
            {"segment":[(84,8), (84, 16)]},
            {"segment":[(84, 16), (0, 16)]},
            {"segment":[(0, 16), (0, 8)]},
    ],
    "entrance":[{ "p1": (0, 9), "p2": (0, 15)},{ "p1": (84, 9), "p2": (84, 15)}],
    "shape":{"p1": (0, 7), "p2": (84, 17)}, # the coordinates of two corners of the scenario map
}

if __name__ == "__main__":
    path_dict = {"1-1": scen_map_1_1}
    for k, v in path_dict.items():
        json_filename = f"data/simulation_data/Scenario{k}/scenario_map.json"
        # Save dictionary to JSON file
        save_dict_to_json(v, json_filename)