import json
import re

# File paths
file1_path = "cleaned_output_v1.json"
file2_path = "alpaca_merged_output.json"
merged_output_path = "alpaca_merged_output_v1.json"

def clean_instruction(instruction):
    if instruction:
        # Remove prefixes like "Q30.6 ", "Q20.1 ", "Q5.2: ", etc.
        return re.sub(r'^Q\d+\.\d+[:\s]+', '', instruction).strip()
    return instruction

# Load the JSON data from both files
with open(file1_path, 'r', encoding='utf-8') as file1:
    data1 = json.load(file1)

with open(file2_path, 'r', encoding='utf-8') as file2:
    data2 = json.load(file2)

# Ensure both files are lists
if not isinstance(data1, list):
    raise ValueError(f"Data in {file1_path} is not a list.")
if not isinstance(data2, list):
    raise ValueError(f"Data in {file2_path} is not a list.")

# Merge the two lists
merged_data = data1 + data2

# Clean and filter entries
filtered_data = []
for entry in merged_data:
    instruction = entry.get("instruction", "")
    cleaned_instruction = clean_instruction(instruction)
    if cleaned_instruction:  # Check if instruction is not empty after cleaning
        entry["instruction"] = cleaned_instruction
        filtered_data.append(entry)

# Write the merged and filtered data to a new JSON file
with open(merged_output_path, 'w', encoding='utf-8') as merged_file:
    json.dump(filtered_data, merged_file, ensure_ascii=False, indent=2)

print(f"Merged and filtered JSON file created at {merged_output_path}")
