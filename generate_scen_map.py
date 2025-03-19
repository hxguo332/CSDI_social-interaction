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
    "entrance":[{ "p1": (0, 9), "p2": (0, 15), "goal_id": 1},{ "p1": (84, 9), "p2": (84, 15), "goal_id": 2}],
    "position":{"p1": (0, 7), "p2": (84, 17)}, # the coordinates of two corners of the scenario map
}

scen_map_2_1 ={
    "obstacles":[], # no obstacles
    "forbidden_area":[
        {"type": "rectangle", "p1": (0, 0), "p2":(30, 30)},
        {"type": "rectangle", "p1": (0, 38), "p2":(22, 70)},
        {"type": "rectangle", "p1": (30, 38), "p2":(70, 70)},
        {"type": "rectangle", "p1": (38, 0), "p2":(70, 30)},
    ],
    "wall":[{ "segment":[(0, 30), (30, 30)]},
            { "segment":[(30, 30), (30, 0)]},
            {"segment":[(30, 0), (38, 0)]},
            {"segment":[(38, 0), (38, 30)]},
            {"segment":[(38, 30), (70, 30)]},
            {"segment":[(70, 30), (70, 38)]},
            {"segment":[(70, 38), (30, 38)]},
            {"segment":[(30, 38), (30, 70)]},
            {"segment":[(30, 70), (22, 70)]},
            {"segment":[(22, 70), (22, 38)]},
            {"segment":[(22, 38), (0, 38)]},
            {"segment":[(0, 38), (0, 30)]},
    ],
    "entrance":[{ "p1": (0, 30), "p2": (0, 38), "goal_id": 2},
                { "p1": (22, 70), "p2": (30, 70), "goal_id": 1},
                { "p1": (70, 38), "p2": (70, 30), "goal_id": 4},
                { "p1": (38, 0), "p2": (30, 0), "goal_id": 3},
    ],
    "position":{"p1": (0, 0), "p2": (70, 70)}, # the coordinates of two corners of the scenario map
}

scen_map_2_2 = scen_map_2_1.copy()
scen_map_2_2["standing_person"] = [(27, 32), (31, 36)] # (x, y)

scen_map_2_3 = scen_map_2_1.copy()
scen_map_2_3["obstacles"] = [
    {
        "type": "rectangle",
        "p1": (25, 31),
        "p2": (27, 34),
    },
    {
        "type": "rectangle",
        "p1": (30, 37),
        "p2": (32, 38),
    },
]

scen_map_3_1 = {
    "obstacles":[
        {"type": "circle", "center": (36, 36), "radius": 2.5},
    ],
    "forbidden_area":[
        {"type": "rectangle", "p1": (0, 0), "p2":(30, 30)},
        {"type": "rectangle", "p1": (42, 0), "p2":(72, 30)},
        {"type": "rectangle", "p1": (0, 42), "p2":(30, 72)},
        {"type": "rectangle", "p1": (42, 42), "p2":(72, 72)},
    ],
    "wall":[{ "segment":[(0, 30), (30, 30)]},
            { "segment":[(30, 30), (30, 0)]},
            {"segment":[(30, 0), (42, 0)]},
            {"segment":[(42, 0), (42, 30)]},
            {"segment":[(42, 30), (72, 30)]},
            {"segment":[(72, 30), (72, 42)]},
            {"segment":[(72, 42), (42, 42)]},
            {"segment":[(42, 42), (42, 72)]},
            {"segment":[(42, 72), (30, 72)]},
            {"segment":[(30, 72), (30, 42)]},
            {"segment":[(30, 42), (0, 42)]},
            {"segment":[(0, 42), (0, 30)]},
    ],
    "entrance":[{ "p1": (0, 30), "p2": (0, 42), "goal_id": 2},
                { "p1": (30, 72), "p2": (42, 72), "goal_id": 1},
                { "p1": (72, 42), "p2": (72, 30), "goal_id": 4},
                { "p1": (42, 0), "p2": (30, 0), "goal_id": 3},
    ],
    "position":{"p1": (0, 0), "p2": (72, 72)}, # the coordinates of two corners of the scenario map

}

scen_map_3_2 = scen_map_3_1.copy()
scen_map_3_2["standing_person"] = [(27, 33.5), (33, 44), (44, 40), (50, 32)] # (x, y)

scen_map_4_1 = {
    "obstacles":[] , # no obstacles
    "forbidden_area":[
        {"type": "polygon", "points":[(20, 28.4), (20, 0), (57.2, 0), (57.2, 36), (26, 18), (20, 28.4)]},
        {"type": "polygon", "points":[(69.2, 0), (69.2, 36), (100.4, 18), (106.4, 28.4), (106.4, 0), (69.2, 0)]},
        {"type": "polygon", "points":[(106.4, 28.4), (75.2, 46.4), (106.4, 64.4), (106.4, 28.4)]},
        {"type": "polygon", "points":[(106.4, 64.4), (106.4, 93), (69.2, 93), (69.2, 56.8), (100.4, 74.8), (106.4, 64.4)]},
        {"type": "polygon", "points":[(57.2, 93), (57.2, 56.8), (26, 74.8), (20, 64.4), (20, 93), (57.2, 93)]},
        {"type": "polygon", "points":[(20, 64.4), (20, 28.4), (51.2, 46.4), (20, 64.4)]},
        {"type": "rectangle", "p1": (106.4, 0), "p2": (107, 93)},
    ],
    "wall":[{ "segment":[(20, 28.4), (26, 18)]},
            { "segment":[(26, 18), (57.2, 36)]},
            {"segment":[(57.2, 36), (57.2, 0)]},
            {"segment":[(57.2, 0), (69.2, 0)]},
            {"segment":[(69.2, 0), (69.2, 36)]},
            {"segment":[(69.2, 36), (100.4, 18)]},
            {"segment":[(100.4, 18), (106.4, 28.4)]},
            {"segment":[(106.4, 28.4), (75.2, 46.4)]},
            {"segment":[(75.2, 46.4), (106.4, 64.4)]},
            {"segment":[(106.4, 64.4), (100.4, 74.8)]},
            {"segment":[(100.4, 74.8), (69.2, 56.8)]},
            {"segment":[(69.2, 56.8), (69.2, 93)]},
            {"segment":[(69.2, 93), (57.2, 93)]},
            {"segment":[(57.2, 93), (57.2, 56.8)]},
            {"segment":[(57.2, 56.8), (26, 74.8)]},
            {"segment":[(26, 74.8), (20, 64.4)]},
            {"segment":[(20, 64.4), (51.2, 46.4)]},
            {"segment":[(51.2, 46.4), (20, 28.4)]},           
    ],
    "entrance":[{ "p1": (20, 64.4), "p2": (26, 74.8), "goal_id": 1},
                { "p1": (57.2, 93), "p2": (69.2, 93), "goal_id": 2},
                { "p1": (106.4, 64.4), "p2": (100.4, 74.8), "goal_id": 3},
                { "p1": (100.4, 18), "p2": (106.4, 28.4), "goal_id": 4},
                { "p1": (69.2, 0), "p2": (57.2, 0), "goal_id": 5},
                { "p1": (26, 18), "p2": (20, 28.4), "goal_id": 6},
    ],
    "standing_person": [(51.2, 36), (54, 55), (75.2, 56.8), (85, 37)], # (x, y)
    "position":{"p1": (20, 0), "p2": (107, 93)}, # the coordinates of two corners of the scenario map
}

if __name__ == "__main__":
    path_dict = {"1-1": scen_map_1_1, "2-1": scen_map_2_1, "2-2": scen_map_2_2, "2-3": scen_map_2_3, "3-1": scen_map_3_1, "3-2": scen_map_3_2, "4-1": scen_map_4_1}
    for k, v in path_dict.items():
        json_filename = f"data/simulation_data/Scenario{k}/scenario_map.json"
        # Save dictionary to JSON file
        save_dict_to_json(v, json_filename)