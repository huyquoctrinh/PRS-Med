import pandas as pd
import random
import csv

def fix_csv_description(csv_file, output_file):
    # Read the CSV file
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Identify the description column (4th column, index 3)
    fixed_rows = []
    for row in rows:
        if len(row) > 3:  # Ensure the row has at least 4 columns
            # Merge all items after column 3 and keep the last one for column 5
            merged_description = ' '.join(row[3:-1]) if len(row) > 4 else row[3]
            row[3] = merged_description
            row[4:] = [row[-1]]  # Keep the last item for column 5
        fixed_rows.append(row[:5])  # Ensure the row has exactly 5 columns
    
    # Write the fixed rows to a new CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(fixed_rows)
    print(f"Fixed CSV file saved to {output_file}")

def replace_description(csv_file, text_file, output_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Read the lines from the text file
    with open(text_file, 'r') as f:
        lines = f.readlines()
    
    # Strip newline characters from the lines
    lines = [line.strip() for line in lines]
    
    # Replace the "description" column with random lines
    if 'description' in df.columns:
        df['description'] = [random.choice(lines) for _ in range(len(df))]
    else:
        print("The 'description' column is not found in the CSV file.")
        return
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
    print(f"Updated CSV file saved to {output_file}")

# fix_csv_description('brain_tumors_ct_scan.csv', 'brain_tumors_ct_scan_fixed.csv')
# fix_csv_description('breast_tumors_ct_scan.csv', 'breast_tumors_ct_scan_fixed.csv')
# fix_csv_description('lung_CT.csv', 'lung_CT_fixed.csv')
# fix_csv_description('lung_Xray.csv', 'lung_Xray_fixed.csv')

# replace_description('brain_tumors_ct_scan_fixed.csv', 'descriptions/brain_tumors_ct_scan.txt', 'brain_tumors_ct_scan_2.csv')

# replace_description('breast_tumors_ct_scan_fixed.csv', 'descriptions/breast_tumors_ct_scan.txt', 'breast_tumors_ct_scan_2.csv')

# replace_description('lung_CT_fixed.csv', 'descriptions/lung_CT.txt', 'lung_CT_2.csv')

# replace_description('lung_Xray_fixed.csv', 'descriptions/lung_Xray.txt', 'lung_Xray_2.csv')
