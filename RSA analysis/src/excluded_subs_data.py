
import pandas as pd

def excluded_subs_data2(excluded_subs: dict, unmatched_subs: dict, data_dict: dict):
    """
    Add unmatched subjects to BOTH 'infant' and 'mom' dictionaries,
    with reason 'Unmatched subject'.
    """

    original_excluded_infants_toys = data_dict['toys']['infant']['refined_best_channel_data']['excluded_subs']
    original_excluded_moms_toys = data_dict['toys']['mom']['refined_best_channel_data']['excluded_subs']
    original_excluded_infants_notoys = data_dict['no_toys']['infant']['refined_best_channel_data']['excluded_subs']
    original_excluded_moms_notoys = data_dict['no_toys']['mom']['refined_best_channel_data']['excluded_subs']



    merged = {}

    for condition, roles in excluded_subs.items():
        merged[condition] = {}
        
        # Copy original infant & mom dicts
        infant_dict = roles.get('infant', {}).copy()
        mom_dict = roles.get('mom', {}).copy()
        
        unmatched_list = unmatched_subs.get(condition, {}).get('unmatched_subjects', [])
        
        for sub_id in unmatched_list:
            # If missing from infant → add unmatched
            if sub_id not in infant_dict:
                infant_dict[sub_id] = "Unmatched subject"
            # If missing from mom → add unmatched
            if sub_id not in mom_dict:
                mom_dict[sub_id] = "Unmatched subject"

        merged[condition]['infant'] = infant_dict
        merged[condition]['mom'] = mom_dict

    rows_toys = []
    rows_no_toys = []

    for condition, roles in merged.items():
        for participant, subjects in roles.items():
            for sub_id, reason in subjects.items():
                row = {
                    "sub_id": sub_id,
                    "participant": participant,
                    "reason": reason
                }
                if condition == "toys":
                    rows_toys.append(row)
                elif condition == "no_toys":
                    rows_no_toys.append(row)

    # Create DataFrames
    toys_df = pd.DataFrame(rows_toys).set_index("sub_id")
    no_toys_df = pd.DataFrame(rows_no_toys).set_index("sub_id")

    # return toys_df_infant, toys_df_mom, no_toys_df_infant, no_toys_df_mom



def excluded_subs_data(excluded_subs: dict, unmatched_subs: dict, data_dict: dict):
    """
    Merge original excluded subjects with excluded_subs + unmatched_subs,
    and return 4 separate DataFrames:
    - toys_df_infant
    - toys_df_mom
    - no_toys_df_infant
    - no_toys_df_mom

    Index: sub_id
    Columns: participant, reason
    """

    # Get original excluded from data_dict
    original_excluded_infants_toys = data_dict['toys']['infant']['refined_best_channel_data']['excluded_subs']
    original_excluded_moms_toys = data_dict['toys']['mom']['refined_best_channel_data']['excluded_subs']
    original_excluded_infants_notoys = data_dict['no_toys']['infant']['refined_best_channel_data']['excluded_subs']
    original_excluded_moms_notoys = data_dict['no_toys']['mom']['refined_best_channel_data']['excluded_subs']

    merged = {}

    for condition, roles in excluded_subs.items():
        merged[condition] = {}

        # Merge original excluded with new excluded_subs
        if condition == 'toys':
            infant_dict = {**original_excluded_infants_toys, **roles.get('infant', {})}
            mom_dict = {**original_excluded_moms_toys, **roles.get('mom', {})}
        else:  # no_toys
            infant_dict = {**original_excluded_infants_notoys, **roles.get('infant', {})}
            mom_dict = {**original_excluded_moms_notoys, **roles.get('mom', {})}

        unmatched_list = unmatched_subs.get(condition, {}).get('unmatched_subjects', [])

        # Add unmatched subjects safely
        for sub_id in unmatched_list:
            if sub_id not in infant_dict:
                infant_dict[sub_id] = "Unmatched subject"
            if sub_id not in mom_dict:
                mom_dict[sub_id] = "Unmatched subject"

        merged[condition]['infant'] = infant_dict
        merged[condition]['mom'] = mom_dict

    # Convert each merged dict to separate DataFrames
    def dict_to_df(sub_dict):
        num_subs = len(sub_dict)
        col_name = f"sub_id ({num_subs} subs)"  # dynamic column name
        rows = []

        for sub_id, reason in sub_dict.items():
            rows.append({col_name: sub_id, "reason": reason})

        df = pd.DataFrame(rows).set_index(col_name)
        return df

    toys_df_infant = dict_to_df(merged['toys']['infant'])
    toys_df_mom = dict_to_df(merged['toys']['mom'])
    no_toys_df_infant = dict_to_df(merged['no_toys']['infant'])
    no_toys_df_mom = dict_to_df(merged['no_toys']['mom'])

    return toys_df_infant, toys_df_mom, no_toys_df_infant, no_toys_df_mom
