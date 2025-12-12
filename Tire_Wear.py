"""
Richard Murray
COMSC 230 Final Project
"""

import os
import pandas as pd

#Set directories
homeFolder = "C:/COMSC/230/Project/Full telemetry"
outputFile = os.path.join(homeFolder, "Tire Wear.csv")

def process_folder(folderPath, outputFile):
    """
    Extracts tire wear data from the last row of each CSV file in the folder
    and creates a new CSV file.
    """
    tireWearList = []
    stintIDs = []
    columns = [
        "LFwearL", "LFwearM", "LFwearR",
        "LRwearL", "LRwearM", "LRwearR",
        "RFwearL", "RFwearM", "RFwearR",
        "RRwearL", "RRwearM", "RRwearR"
    ]

    #Get all CSV files in the folder
    csvFiles = [file for file in os.listdir(folderPath) if file.endswith(".csv")]

    for file in csvFiles:
        filePath = os.path.join(folderPath, file)

        #Read the CSV file
        df = pd.read_csv(filePath)

        #Extract the last row
        lastRow = df.iloc[-1]

        #Extract the required columns
        tireWearRow = lastRow[columns].values
        tireWearList.append(tireWearRow)
        #Use the file name as the stint ID
        stintIDs.append(file)  

    #Create a DataFrame for the new CSV file
    tireWear_df = pd.DataFrame(tireWearList, columns=columns, index=stintIDs)
    tireWear_df.index.name = "Stint ID"

    #Save the new CSV file
    tireWear_df.to_csv(outputFile, index=True)
    print(f"Tire wear data saved to {outputFile}")

#Run the function
process_folder(homeFolder, outputFile)