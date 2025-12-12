"""
Richard Murray
COMSC 230 Final Project
"""

import pandas as pd
import os

#Set directories
homeFolder = "C:/COMSC/230/Project/Laps"
inputFile = "c:/COMSC/230/Project/Laps/Max_Lap_Times.csv"

def summarize_lap_times(inputFile, outputFile):
    """
    Summarizes lap time data by calculating the fastest lap, best 5 laps average,
    and best 10 laps average for each stint.
    """
    #Read CSV file
    df = pd.read_csv(inputFile, index_col=0)

    #List to store summary data
    summaryData = []

    #Process each stint
    for stintID, row in df.iterrows():
        #Convert lap times to a sorted list
        times = sorted(row.tolist())

        #Find fastest lap
        fastest = times[0]

        #Find average of the best 5 laps
        avg5 = sum(times[:5]) / 5

        #Find average of the best 10 laps
        avg10 = sum(times[:10]) / 10

        #Append the summary data for this stint
        summaryData.append({
            "Stint ID": stintID,
            "Fastest Lap": fastest,
            "Best 5 Laps Avg": avg5,
            "Best 10 Laps Avg": avg10
        })

    #New DataFrame for the summary data
    summary_df = pd.DataFrame(summaryData)

    #Save the summary DataFrame to a new CSV file
    summary_df.to_csv(outputFile, index=False)
    print(f"Lap time summary saved to {outputFile}")

#Run the function
outputFile = os.path.join(homeFolder, "Lap_Time_Targets.csv")
summarize_lap_times(inputFile, outputFile)