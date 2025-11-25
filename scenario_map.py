import numpy as np
import cv2

def world_to_grid(x, y, position, height, scale):
    """
    Convert world coordinates to grid indices.
    
    - Adjusts coordinates based on the bottom-left origin.
    - Scales up by the given scale factor.

    Args:
        x (int): X-coordinate in world space.
        y (int): Y-coordinate in world space.
        position (dict): Bounding box of the scenario.
        height (int): Height of the world in grid units.
        scale (int): Scaling factor (1 world unit = scale pixels).

    Returns:
        (int, int): Scaled grid coordinates.
    """
    grid_x = (x - position["p1"][0]) * scale
    grid_y = (height - (y - position["p1"][1])) * scale  # Flip Y-axis and scale
    return grid_x, grid_y

def create_scenario_map(scenario, scale=10, ignroe_standing_person=True):
    """
    Generates a high-resolution grid map based on the scenario definition.
    
    **Color Legend:**
    - `128 (Gray)`: Walkable area
    - `0 (Black)`: Walls, obstacles, forbidden areas
    - `255 (White)`: Entrances (walkable openings)
    - `200 (Light Gray)`: Standing persons
    - `0 (Black Circles)`: Circular obstacles
    
    Args:
        scenario (dict): Scenario definition containing obstacles, walls, etc.
        scale (int, optional): Resolution scaling factor (default=10).

    Returns:
        np.array: Generated map as a NumPy array.
    """
    # Get the bounding box of the shape (scaled)
    width = (scenario["position"]["p2"][0] - scenario["position"]["p1"][0])
    height = (scenario["position"]["p2"][1] - scenario["position"]["p1"][1])
    grid = np.full((height * scale, width * scale), 128, dtype=np.uint8)  # Default walkable (128)
    
    # Draw forbidden areas (0)
    for area in scenario["forbidden_area"]:
        if area["type"] == "rectangle":
            x1, y1 = world_to_grid(*area["p1"], scenario["position"], height, scale)
            x2, y2 = world_to_grid(*area["p2"], scenario["position"], height, scale)
            x_min, x_max = min(x1, x2), max(x1, x2)  # Ensure proper bounding
            y_min, y_max = min(y1, y2), max(y1, y2)
            grid[int(y_min):int(y_max)+1, int(x_min):int(x_max)+1] = 0  # Fill area with black (0)
        elif area["type"] == "polygon":
            pts = np.array([world_to_grid(*pt, scenario["position"], height, scale) for pt in area["points"]], np.int32)
            cv2.fillPoly(grid, [pts], 0)

    # Draw obstacles (0)
    for obstacle in scenario["obstacles"]:
        if obstacle["type"] == "rectangle":
            x1, y1 = world_to_grid(*obstacle["p1"], scenario["position"], height, scale)
            x2, y2 = world_to_grid(*obstacle["p2"], scenario["position"], height, scale)
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            grid[int(y_min):int(y_max)+1, int(x_min):int(x_max)+1] = 0  # Fill area with black (0)
        elif obstacle["type"] == "circle":
            cx, cy = world_to_grid(*obstacle["center"], scenario["position"], height, scale)
            radius = obstacle["radius"] * scale  # Scale the radius
            cv2.circle(grid, (int(cx), int(cy)), int(radius), 0, -1)  # Fill circle with black (0)

    # Draw walls (0)
    for wall in scenario["wall"]:
        (x1, y1), (x2, y2) = wall["segment"]
        x1, y1 = world_to_grid(x1, y1, scenario["position"], height, scale)
        x2, y2 = world_to_grid(x2, y2, scenario["position"], height, scale)
        cv2.line(grid, (int(x1), int(y1)), (int(x2), int(y2)), 0, 2)  # Thicker walls

    # Draw entrances (255)
    for entrance in scenario["entrance"]:
        x1, y1 = world_to_grid(*entrance["p1"], scenario["position"], height, scale)
        x2, y2 = world_to_grid(*entrance["p2"], scenario["position"], height, scale)
        cv2.line(grid, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)  # Thicker entrance

    # Draw standing persons (200)
    if not ignroe_standing_person:
        for person in scenario["standing_person"]:
            x, y = world_to_grid(*person, scenario["position"], height, scale)
            cv2.circle(grid, (int(x), int(y)), scale//3, 200, -1)  # Draw person as a small circle

    return grid

def save_map(grid, filename="scenario_map_highres.png"):
    """Saves the generated map as an image file."""
    cv2.imwrite(filename, grid)
    print(f"Map saved as {filename}")

def create_scenario_map_3channel_with_last_channel_empty(scenario, scale=10, draw_standing_person=False):
    """
    Generates a 3-channel high-resolution scenario map.
    
    **Channel Definitions:**
    - **Channel 1 (Red) 🟥**: Obstacles (rectangles, circles) & optionally standing persons & Walls & forbidden areas
    - **Channel 2 (Green) 🟩**: Entrances
    - **Channel 3 (Blue) 🟦**: Empty

    Args:
        scenario (dict): Scenario definition containing obstacles, walls, entrances, etc.
        scale (int, optional): Resolution scaling factor (default=10).
        draw_standing_person (bool, optional): Whether to include standing persons in channel 1.
    Returns:
        np.array: Generated 3-channel scenario map as a NumPy array.
    """
    # Build the standard 3-channel map first
    base = create_scenario_map_3channel(
        scenario, scale=scale, draw_standing_person=draw_standing_person
    )

    # Allocate output map
    out = np.zeros_like(base)

    # Merge obstacles (ch0) and walls/forbidden (ch1) into channel 0
    out[:, :, 0] = np.maximum(base[:, :, 0], base[:, :, 1])

    # Move entrances (original channel 2) into channel 1 (green)
    out[:, :, 1] = base[:, :, 2]

    # Leave last channel (2) empty (zeros)
    return out

def create_scenario_map_3channel(scenario, scale=10, draw_standing_person=False):
    """
    Generates a 3-channel high-resolution scenario map.
    
    **Channel Definitions:**
    - **Channel 1 (Red) 🟥**: Obstacles (rectangles, circles) & optionally standing persons
    - **Channel 2 (Green) 🟩**: Walls & forbidden areas
    - **Channel 3 (Blue) 🟦**: Entrances

    Args:
        scenario (dict): Scenario definition containing obstacles, walls, entrances, etc.
        scale (int, optional): Resolution scaling factor (default=10).
        draw_standing_person (bool, optional): Whether to include standing persons in channel 1.

    Returns:
        np.array: Generated 3-channel scenario map as a NumPy array.
    """
    width = (scenario["position"]["p2"][0] - scenario["position"]["p1"][0])
    height = (scenario["position"]["p2"][1] - scenario["position"]["p1"][1])
    
    # Create a 3-channel blank map (default all black)
    grid = np.zeros((height * scale, width * scale, 3), dtype=np.uint8)

    #print(f"Generated 3-channel map shape: {grid.shape} (Scale: {scale}x)")

    # Draw forbidden areas (Green, Channel 2)
    for area in scenario["forbidden_area"]:
        if area["type"] == "rectangle":
            x1, y1 = world_to_grid(*area["p1"], scenario["position"], height, scale)
            x2, y2 = world_to_grid(*area["p2"], scenario["position"], height, scale)
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            grid[int(y_min):int(y_max)+1, int(x_min):int(x_max)+1, 1] = 255  # Green (Channel 2)
        elif area["type"] == "polygon":
            pts = np.array([world_to_grid(*pt, scenario["position"], height, scale) for pt in area["points"]], np.int32)
            cv2.fillPoly(grid, [pts], (0, 255, 0))  # Green (Channel 2)

    # Draw obstacles (Red, Channel 1)
    for obstacle in scenario["obstacles"]:
        if obstacle["type"] == "rectangle":
            x1, y1 = world_to_grid(*obstacle["p1"], scenario["position"], height, scale)
            x2, y2 = world_to_grid(*obstacle["p2"], scenario["position"], height, scale)
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            grid[int(y_min):int(y_max)+1, int(x_min):int(x_max)+1, 0] = 255  # Red (Channel 1)
        elif obstacle["type"] == "circle":
            cx, cy = world_to_grid(*obstacle["center"], scenario["position"], height, scale)
            radius = obstacle["radius"] * scale
            cv2.circle(grid, (int(cx), int(cy)), int(radius), (255, 0, 0), -1)  # Red (Channel 1)

    # Draw walls (Green, Channel 2)
    for wall in scenario["wall"]:
        (x1, y1), (x2, y2) = wall["segment"]
        x1, y1 = world_to_grid(x1, y1, scenario["position"], height, scale)
        x2, y2 = world_to_grid(x2, y2, scenario["position"], height, scale)
        cv2.line(grid, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green (Channel 2)

    # Draw entrances (Blue, Channel 3)
    for entrance in scenario["entrance"]:
        x1, y1 = world_to_grid(*entrance["p1"], scenario["position"], height, scale)
        x2, y2 = world_to_grid(*entrance["p2"], scenario["position"], height, scale)
        cv2.line(grid, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Blue (Channel 3)

    # Draw standing persons (Red, Channel 1) if enabled
    if draw_standing_person:
        for person in scenario["standing_person"]:
            x, y = world_to_grid(*person, scenario["position"], height, scale)
            cv2.circle(grid, (int(x), int(y)), scale//3, (255, 0, 0), -1)  # Red (Channel 1)

    return grid

def save_map_3channel(grid, filename="scenario_map_3channel.png"):
    """Saves the generated 3-channel map as an image file."""
    cv2.imwrite(filename, grid)
    print(f"3-Channel Map saved as {filename}")

def display_map_3channel(grid):
    import matplotlib.pyplot as plt
    """Displays the generated 3-channel scenario map using matplotlib."""
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
    plt.title("3-Channel Scenario Map")
    plt.show()

if __name__ == "__main__":
    import json
    # Sample Scenario Definition
    scen_map_example = {
        "obstacles": [
            {"type": "rectangle", "p1": (30, 11), "p2": (28, 9)},  # Now works even if p1 & p2 are reversed!
            {"type": "rectangle", "p1": (58, 13), "p2": (54, 12)},
            {"type": "circle", "center": (40, 10), "radius": 2},  # New Circular Obstacle
        ],
        "forbidden_area": [
            {"type": "polygon", "points": [(0, 7), (0, 8), (84, 8), (84, 7)]},
            {"type": "rectangle", "p1": (84, 17), "p2": (0, 16)},  # Now supports any order!
        ],
        "standing_person": [(34, 14), (61, 10)],  # (x, y)
        "wall": [
            {"segment": [(0, 8), (84, 8)]},
            {"segment": [(84, 8), (84, 16)]},
            {"segment": [(84, 16), (0, 16)]},
            {"segment": [(0, 16), (0, 8)]},
        ],
        "entrance": [
            {"p1": (0, 9), "p2": (0, 15)},
            {"p1": (84, 9), "p2": (84, 15)},
        ],
        "position": {"p1": (0, 7), "p2": (84, 17)},
    }
    scen = '3-1'
    path = f"data/simulation_data/Scenario{scen}/scenario_map.json"
    scen_map_example = json.load(open(path, "r"))

    scale_factor = 10
    grid_map = create_scenario_map(scen_map_example, scale=scale_factor)
    save_map(grid_map, "scenario_map_highres.png")

    # Generate and save the 3-channel scenario map
    grid_map_3channel = create_scenario_map_3channel(
        scen_map_example, scale=scale_factor, draw_standing_person=False
    )
    save_map_3channel(grid_map_3channel, f"scenario_{scen}_map_3channel.png")

    # Generate and save the merged-first-two-channels variant (last channel empty)
    grid_map_merged = create_scenario_map_3channel_with_last_channel_empty(
        scen_map_example, scale=scale_factor, draw_standing_person=False
    )
    # Simple sanity check: ensure last channel is empty
    assert (grid_map_merged[:, :, 2] == 0).all(), "Last channel should be empty"
    save_map_3channel(grid_map_merged, f"scenario_{scen}_map_3channel_last_empty.png")
    #display_map_3channel(grid_map_3channel)
