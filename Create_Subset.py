"""
Richard Murray
COMSC 230 Final Project
"""

import numpy as np
from itertools import product
import pandas as pd
"""
Define a list for each factor indicating the number of levels
Order:     Left Air Pressure, 
           Right Air Pressure, 
           Rear Stagger, 
           Front Stagger,
           Front Spring Rate, 
           Rear Spring Rate, 
           Ride Height,
           Left Front Camber, 
           Right Front Camber, 
           Front ARB Preload
"""
def main():
    """
    Generates a subset of points from a full factorial matrix
    by maximizing the minimum distance between points.
    """
    #Number of levels for each factor
    factor_levels = [3, 3, 2, 2, 3, 3, 2, 2, 2, 2]

    #Generate the full factorial matrix of every combination
    full_design = np.array(list(product(*[range(level) for level in factor_levels])))

    print("Shape of full matrix:", full_design.shape)
    print("Full matrix:\n", full_design)

    #Number of samples to select
    n_samples = 100

    #Initialize the subset design with the first point
    np.random.seed(43)
    subset_indices = [np.random.choice(len(full_design))]
    subset_design = [full_design[subset_indices[0]]]

    #Iteratively select points to maximize the minimum distance between points
    for _ in range(1, n_samples):
        distances = np.min(
            np.linalg.norm(full_design[:, None, :] - np.array(subset_design)[None, :, :], axis=2),
            axis=1
        )
        next_index = np.argmax(distances)
        subset_indices.append(next_index)
        subset_design.append(full_design[next_index])

    subset_design = np.array(subset_design)

    print("Shape of subset design matrix:", subset_design.shape)
    print("Subset design matrix:\n", subset_design)

    #Map the subset design matrix to actual values
    mapped_subset_design = map_design(subset_design)
    print("Mapped subset design matrix:\n", mapped_subset_design)

    #Save the mapped subset design to a CSV
    save_to_csv(mapped_subset_design)

#Define mappings for each factor
mappings = {
    "left_air_pressure": {0: 13, 1: 20, 2: 27},
    "right_air_pressure": {0: 23, 1: 32, 2: 41},
    "rear_stagger": {0: 0.25, 1: 1.25},
    "front_stagger": {0: 0.125, 1: 0.625},
    "front_spring_rate": {0: 700, 1: 1600, 2: 2300},
    "rear_spring_rate": {0: 200, 1: 300, 2: 400},
    "ride_height": {0: 5.75, 1: 6.25},
    "left_front_camber": {0: 0.5, 1: 3.5},
    "right_front_camber": {0: -3.5, 1: -4.5},
    "front_arb_preload": {0: 30, 1: -30},
}

def map_code(factor_name, code):
    
    #Calls on dictionary to map the code to the actual value
    return mappings[factor_name][int(code)]

def map_design(design_matrix):
    
    #Maps the matrix of simple values to the actual values
    factor_names = [
        "left_air_pressure", "right_air_pressure", "rear_stagger",
        "front_stagger", "front_spring_rate", "rear_spring_rate",
        "ride_height", "left_front_camber", "right_front_camber",
        "front_arb_preload"
    ]
    mapped_design = []
    for row in design_matrix:
        mapped_row = [map_code(factor_names[i], row[i]) for i in range(len(row))]
        mapped_design.append(mapped_row)
    return np.array(mapped_design)

def save_to_csv(mapped_subset_design, filename="MiniStock_Setups.csv"):
    
    #Saves the final matrix to a CSV
    df = pd.DataFrame(mapped_subset_design, columns=[
        "Left Air Pressure", "Right Air Pressure", "Rear Stagger",
        "Front Stagger", "Front Spring Rate", "Rear Spring Rate",
        "Ride Height", "Left Front Camber", "Right Front Camber",
        "Front ARB Preload"
    ])
    df.to_csv(filename, index=False)
    print(f"Subset design saved to {filename}")

if __name__ == "__main__":
    main()