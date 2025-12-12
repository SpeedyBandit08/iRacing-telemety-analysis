"""
Richard Murray
COMSC 230 Final Project
"""

import os
import pandas as pd

#Set directories
homeFolder = "C:/COMSC/230/Project/15 Lap telemetry"
os.chdir(homeFolder)

def process_folder(folderPath, outputFile):
    """
    Processes all CSV files in the folder, finds the maximum value in the 'LapCurrentLapTime'
    column for each lap in the 'Lap' column, and saves the results to a new DataFrame.
    Columns represent individual laps, and rows represent each stint ID.
    """
    lapTimeList = []
    stintIDs = []

    #Get all CSV files in the folder
    csvFiles = [file for file in os.listdir(folderPath) if file.endswith(".csv")]

    for file in csvFiles:
        filePath = os.path.join(folderPath, file)

        #Read the CSV
        df = pd.read_csv(filePath)

        #Find the maximum value in the 'LapCurrentLapTime' column for each lap
        maxLapTimes = df.groupby('Lap')['LapCurrentLapTime'].max().reset_index()

        #Convert the maximum lap times to a list
        lapTimes = maxLapTimes['LapCurrentLapTime'].tolist()

        #Append the lap times and stint ID (file name)
        lapTimeList.append(lapTimes)
        stintIDs.append(file) 
        #debugging
        #print(f"Processed file: {file}")

    #Make sure the same number of laps are in each file
    maxLaps = max(len(times) for times in lapTimeList)

    #Name the columns based on the lap count
    columns = [f"Lap {i+1}" for i in range(maxLaps)]

    #Create a DataFrame for the new CSV file
    lapTime_df = pd.DataFrame(lapTimeList, columns=columns, index=stintIDs)
    lapTime_df.index.name = "Stint ID" 

    #Save the new CSV file
    lapTime_df.to_csv(outputFile, index=True)
    print(f"Lap time data saved to {outputFile}")

#Run the function and name the file
outputFile = os.path.join(homeFolder, "Max_Lap_Times.csv")
process_folder(homeFolder, outputFile)