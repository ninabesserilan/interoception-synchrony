import pickle
from pathlib import Path


def filter_group_condition(data, group, condition, participant):
    """
    Extract all data for a given group (age) and condition.
    
    Args:
        data (dict): The full ibis_data dict.
        group (str): e.g., "9_months"
        condition (str): e.g., "toys"
        
    Returns:
        dict: Filtered dictionary {dyad_id -> {participant -> {channel -> data}}}
    """
    if group not in data:
        raise KeyError(f"Group {group} not found")

    filtered = {}
    for dyad_id, conds in data[group].items():
        if condition not in conds:
            continue

        cond_data = conds[condition]

        if participant:
            if participant in cond_data:
                filtered[dyad_id] = {participant: cond_data[participant]}

        else:

            raise KeyError(f"Participants not found")
        
    filtered = dict(sorted(filtered.items(), key=lambda x: int(x[0])))

    return filtered


# ---- Load Pickles ---------------------------

# pickle_path_9m_ibis = Path("C:\\Users\\ninab36\\python code\\Files data\\01_ibi_after_extraction_ibis_data.pkl")
pickle_path_9m_ibis = Path('/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Files data/01_ibi_after_extraction_ibis_data.pkl')

# pickle_path_9_month_peaks = Path("C:\\Users\\ninab36\\python code\\Files data\\01_ibi_after_extraction_peaks_data.pkl")
pickle_path_9_month_peaks = Path('/Users/nina/Desktop/University of Vienna/PhD projects/python code/interoception-synchrony/Files data/01_ibi_after_extraction_peaks_data.pkl')

with open(pickle_path_9m_ibis, "rb") as f_ibis:
    ibis_data_9m = pickle.load(f_ibis)

with open(pickle_path_9_month_peaks, "rb") as f_peaks:
    peaks_data_9m = pickle.load(f_peaks)

# ---- Data Structure ---------------------------

# 9 months - ibis
ibis_toys_9m_infants_data = filter_group_condition(ibis_data_9m, group="9_months", condition="toys", participant= "infant")
ibis_toys_9m_moms_data = filter_group_condition(ibis_data_9m, group="9_months", condition="toys", participant= "mom")

ibis_no_toys_9m_infants_data = filter_group_condition(ibis_data_9m, group="9_months", condition="no_toys", participant= "infant")
ibis_no_toys_9m_moms_data = filter_group_condition(ibis_data_9m, group="9_months", condition="no_toys", participant= "mom")

# 9 months - peaks
peaks_toys_9m_infants_data = filter_group_condition(peaks_data_9m, group="9_months", condition="toys", participant= "infant")
peaks_toys_9m_moms_data = filter_group_condition(peaks_data_9m, group="9_months", condition="toys", participant= "mom")

peaks_no_toys_9m_infants_data = filter_group_condition(peaks_data_9m, group="9_months", condition="no_toys", participant= "infant")
peaks_no_toys_9m_moms_data = filter_group_condition(peaks_data_9m, group="9_months", condition="no_toys", participant= "mom")
