import os
import pandas as pd
import json

# Folder containing the Excel files
folder_path = "cleaned_question_pair"
output_json_path = "cleaned_output.json"

# List to hold the formatted JSON data
data = []

# Loop through all Excel files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        # Read the Excel file
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path, header=None)  # No headers in the files

        # Iterate through each row and format data
        for index, row in df.iterrows():
            instruction = row[0] if not pd.isnull(row[0]) else ""
            output = row[1] if not pd.isnull(row[1]) else ""
            formatted_entry = {
                "instruction": instruction,
                "input": "",
                "output": output,
                "system": "You're a helpful assistant in the ... domain."
            }
            data.append(formatted_entry)

# Write the list to a JSON file
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=2)

print(f"JSON file created successfully at {output_json_path}")
