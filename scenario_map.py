import numpy as np
import cv2

def world_to_grid(x, y, shape, height, scale):
    """
    Convert world coordinates to grid indices.
    
    - Adjusts coordinates based on the bottom-left origin.
    - Scales up by the given scale factor.

    Args:
        x (int): X-coordinate in world space.
        y (int): Y-coordinate in world space.
        shape (dict): Bounding box of the scenario.
        height (int): Height of the world in grid units.
        scale (int): Scaling factor (1 world unit = scale pixels).

    Returns:
        (int, int): Scaled grid coordinates.
    """
    grid_x = (x - shape["p1"][0]) * scale
    grid_y = (height - (y - shape["p1"][1])) * scale  # Flip Y-axis and scale
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
    width = (scenario["shape"]["p2"][0] - scenario["shape"]["p1"][0])
    height = (scenario["shape"]["p2"][1] - scenario["shape"]["p1"][1])
    grid = np.full((height * scale, width * scale), 128, dtype=np.uint8)  # Default walkable (128)
    
    # Draw forbidden areas (0)
    for area in scenario["forbidden_area"]:
        if area["type"] == "rectangle":
            x1, y1 = world_to_grid(*area["p1"], scenario["shape"], height, scale)
            x2, y2 = world_to_grid(*area["p2"], scenario["shape"], height, scale)
            x_min, x_max = min(x1, x2), max(x1, x2)  # Ensure proper bounding
            y_min, y_max = min(y1, y2), max(y1, y2)
            grid[int(y_min):int(y_max)+1, int(x_min):int(x_max)+1] = 0  # Fill area with black (0)
        elif area["type"] == "polygon":
            pts = np.array([world_to_grid(*pt, scenario["shape"], height, scale) for pt in area["points"]], np.int32)
            cv2.fillPoly(grid, [pts], 0)

    # Draw obstacles (0)
    for obstacle in scenario["obstacles"]:
        if obstacle["type"] == "rectangle":
            x1, y1 = world_to_grid(*obstacle["p1"], scenario["shape"], height, scale)
            x2, y2 = world_to_grid(*obstacle["p2"], scenario["shape"], height, scale)
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            grid[int(y_min):int(y_max)+1, int(x_min):int(x_max)+1] = 0  # Fill area with black (0)
        elif obstacle["type"] == "circle":
            cx, cy = world_to_grid(*obstacle["center"], scenario["shape"], height, scale)
            radius = obstacle["radius"] * scale  # Scale the radius
            cv2.circle(grid, (int(cx), int(cy)), int(radius), 0, -1)  # Fill circle with black (0)

    # Draw walls (0)
    for wall in scenario["wall"]:
        (x1, y1), (x2, y2) = wall["segment"]
        x1, y1 = world_to_grid(x1, y1, scenario["shape"], height, scale)
        x2, y2 = world_to_grid(x2, y2, scenario["shape"], height, scale)
        cv2.line(grid, (int(x1), int(y1)), (int(x2), int(y2)), 0, 2)  # Thicker walls

    # Draw entrances (255)
    for entrance in scenario["entrance"]:
        x1, y1 = world_to_grid(*entrance["p1"], scenario["shape"], height, scale)
        x2, y2 = world_to_grid(*entrance["p2"], scenario["shape"], height, scale)
        cv2.line(grid, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)  # Thicker entrance

    # Draw standing persons (200)
    if not ignroe_standing_person:
        for person in scenario["standing_person"]:
            x, y = world_to_grid(*person, scenario["shape"], height, scale)
            cv2.circle(grid, (int(x), int(y)), scale//3, 200, -1)  # Draw person as a small circle

    return grid

def save_map(grid, filename="scenario_map_highres.png"):
    """Saves the generated map as an image file."""
    cv2.imwrite(filename, grid)
    print(f"Map saved as {filename}")

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
        "shape": {"p1": (0, 7), "p2": (84, 17)},
    }
    #path = "data/simulation_data/Scenario1-1/scenario_map.json"
    #scen_map_1_1 = json.load(open(path, "r"))

    scale_factor = 10
    grid_map = create_scenario_map(scen_map_example, scale=scale_factor)
    save_map(grid_map, "scenario_map_highres.png")