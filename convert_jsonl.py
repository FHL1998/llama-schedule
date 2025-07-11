import json

# Input and output file paths
jsonl_path = "fine_tune_data.jsonl"
json_path = "fine_tune_data.json"

# Read the JSONL file and load each line as a JSON object
with open(jsonl_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Write the list of JSON objects to a single JSON file
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
