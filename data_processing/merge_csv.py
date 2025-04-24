import os
import pandas as pd

# Define the label
label = "lung_Xray"  # Replace with your desired label

# File paths
file_fixed = f"{label}_fixed.csv"
file_2 = f"{label}_2.csv"
output_file = f"{label}.csv"

# Read the CSV files
df_fixed = pd.read_csv(file_fixed)
df_2 = pd.read_csv(file_2)

# Rename the 'description' column in the second file to 'question'
df_2 = df_2.rename(columns={"description": "question"})

# Select only the 'question' column from the second DataFrame
df_2 = df_2[['question']]

# Merge the two DataFrames
merged_df = df_fixed.join(df_2)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv(output_file, index=False)

print(f"Merged file saved as {output_file}")