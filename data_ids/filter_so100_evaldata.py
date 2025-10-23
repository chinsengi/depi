import json

# Read the dataset list
with open("./dataset_list_valid.json") as f:
    dataset_list = json.load(f)

# Filter out datasets with "eval" in the name
filtered_list = [dataset_id for dataset_id in dataset_list if "eval" not in dataset_id.lower()]

print(f"Original dataset count: {len(dataset_list)}")
print(f"Filtered dataset count: {len(filtered_list)}")

# Save filtered list back to file
with open("./so100_data/dataset_list_valid.json", "w") as f:
    json.dump(filtered_list, f, indent=4)
