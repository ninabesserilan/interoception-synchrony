import os
import pandas as pd
import pickle
from config import  Config
# Folder containing your CSV files



folder_path = "C:\\Users\\ninab36\\Interoception x Physiological Synchrony\\Analysis\\Data\\ibiData 3"

# List all CSV files first to get total count
all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
total_files = len(all_files)

# Dictionary to store all CSV data
data_dict = {}

for idx, file_name in enumerate(all_files, 1):
    print(f"Processing file {idx}/{total_files}: {file_name}")

    file_path = os.path.join(folder_path, file_name)

    # Load CSV safely
    try:
        csv_data = pd.read_csv(file_path, header=None).squeeze()
    except pd.errors.EmptyDataError:
        print(f"Warning: {file_name} could not be read, skipping.")
        continue

    # Detect file attributes using CONFIG
    file_attributes = {}
    for category, options_dict in Config.items():
        for attribute_name, filename_marker in options_dict.items():
            if filename_marker in file_name:
                file_attributes[category] = attribute_name

                # Extract dyad ID for group
                if category == 'groups':
                    start_index = file_name.find(filename_marker)
                    dyad_id = file_name[start_index + len(filename_marker): start_index + len(filename_marker) + 2]
                    file_attributes['dyad_id'] = dyad_id
                break

    # Assign variables for clarity
    data_type = file_attributes.get('data_type')
    group = file_attributes.get('groups')
    dyad_id = file_attributes.get('dyad_id')
    condition = file_attributes.get('conditions')
    participant = file_attributes.get('participants')
    channel = file_attributes.get('channels')

    # Check all attributes were found
    if not all([data_type, group, dyad_id, condition, participant, channel]):
        print(f"Warning: Could not fully parse {file_name}, skipping.")
        continue

    # Build nested dictionary with placeholder for selected channel
    data_dict.setdefault(data_type, {}) \
             .setdefault(group, {}) \
             .setdefault(dyad_id, {}) \
             .setdefault(condition, {}) \
             .setdefault(participant, {})[channel] = {
                 "data": csv_data,
                 "selected_channel": None  # placeholder for your later choice
             }

# Save to pickle
with open("all_data.pkl", "wb") as pickle_file:
    pickle.dump(data_dict, pickle_file)

print(f"All CSV data has been saved to all_data.pkl ({total_files} files processed).")


# Load the full pickle (or if you still have `data_dict` in memory, skip this)
with open("all_data.pkl", "rb") as pickle_file:
    data_dict = pickle.load(pickle_file)

# Save only ibis data
if 'ibis' in data_dict:
    with open("ibis_data.pkl", "wb") as f_ibis:
        pickle.dump(data_dict['ibis'], f_ibis)
    print("Saved ibis data to ibis_data.pkl")

# Save only peaks data
if 'peaks' in data_dict:
    with open("peaks_data.pkl", "wb") as f_peaks:
        pickle.dump(data_dict['peaks'], f_peaks)
    print("Saved peaks data to peaks_data.pkl")
