"""
Richard Murray
COMSC 230 Final Project
"""

import os
import pandas as pd

#Set directories
homeFolder = "C:/COMSC/230/Project/Original telemetry"

def process_folder(folderPath):
    """
    Processes all files in the folder. For CSV files, skips the last 9 rows
    and saves the cleaned data to a new file.
    """
    #Iterate through all files in the folder
    for file in os.listdir(folderPath):
        filePath = os.path.join(folderPath, file)

        #Check if the file is a CSV
        if file.endswith(".csv"):
            #Read the CSV file, skipping the last 9 rows
            cleanCSV = pd.read_csv(filePath, skiprows=9)
            savePath = os.path.join(folderPath, file.replace(".csv", "_clean.csv"))

            #Save the cleaned CSV to a new file
            cleanCSV.to_csv(savePath, index=False)
            print(f"Cleaned CSV saved to {savePath}")

        #Skip .ibt files
        elif file.endswith(".ibt"):
            continue

#Run the function
process_folder(homeFolder)