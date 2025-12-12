"""
Richard Murray
COMSC 230 Final Project
"""

import pandas as pd
import os

#Set directories
homeFolder = "C:/COMSC/230/Project/Training"
tireWearFile = "C:/COMSC/230/Project/Tires/Tire Wear.csv"
lapTimeTargetsFile = "C:/COMSC/230/Project/Laps/Lap_Time_Targets.csv"
miniStockSetupsFile = "C:/COMSC/230/Project/Setups/MiniStock_Setups.csv"
outputFile = "C:/COMSC/230/Project/Training/Combined_Data.csv"

def combine_data(tireWearFile, lapTimeTargetsFile, miniStockSetupsFile, outputFile):
    """
    Combines data from the three CSV files horizontally, matching the indexes of the setups to stint ID.
    The indexes in MiniStock_Setups.csv are already correct, but the indexes in the other two files
    need to be converted from strings to integers.
    """
    #Load the CSV files into DataFrames
    tireWear_df = pd.read_csv(tireWearFile)
    lapTimeTargets_df = pd.read_csv(lapTimeTargetsFile)
    miniStockSetups_df = pd.read_csv(miniStockSetupsFile)

    #Extract numeric values from the 'Stint ID' column of tire wear and lap time targets
    #The index I'm looking for always follows the characters 'id' in the string
    tireWear_df['Stint ID'] = tireWear_df['Stint ID'].str.extract(r'id(\d+)').astype(int)
    lapTimeTargets_df['Stint ID'] = lapTimeTargets_df['Stint ID'].str.extract(r'id(\d+)').astype(int)

    #Sort the DataFrames by the numeric value of the 'Stint ID'
    tireWear_df.sort_values('Stint ID', inplace=True)
    lapTimeTargets_df.sort_values('Stint ID', inplace=True)

    #Reset the index to align with MiniStock_Setups
    tireWear_df.reset_index(drop=True, inplace=True)
    lapTimeTargets_df.reset_index(drop=True, inplace=True)

    #Combine the data horizontally
    combined_df = pd.concat([miniStockSetups_df, tireWear_df, lapTimeTargets_df], axis=1)

    #Drop duplicate Stint ID column and make a new one
    combined_df.drop(columns=['Stint ID'], inplace=True)
    combined_df.index.name = 'Stint ID'

    #Save the combined DataFrame to a new CSV file
    os.makedirs(os.path.dirname(outputFile), exist_ok=True)
    combined_df.to_csv(outputFile, index=True)
    print(f"Combined data saved to {outputFile}")

#Run the function
combine_data(tireWearFile, lapTimeTargetsFile, miniStockSetupsFile, outputFile)