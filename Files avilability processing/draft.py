# import pandas as pd
# import os

# def data_availability_df(data_dict):
#     """
#     Returns a DataFrame with dyad_id as index.
#     Columns: condition_participant_channel (e.g., with_toys_infant_ch_0).
#     Values: True/False depending on whether data exists.
#     """

#     availability = {}

#     for group_name, group_data in data_dict.items():
#         # Map group to suffix (9 or 18)
#         group_suffix = "9" if "9_months" in group_name else "18"

#         for dyad_id, dyad_data in group_data.items():
#             if dyad_id not in availability:
#                 availability[dyad_id] = {col: "False" for col in [
#                     f"infant_9_ch_0", f"infant_9_ch_1", f"infant_9_ch_2",
#                     f"mom_9_ch_0", f"mom_9_ch_1", f"mom_9_ch_2",
#                     f"infant_18_ch_0", f"infant_18_ch_1", f"infant_18_ch_2",
#                     f"mom_18_ch_0", f"mom_18_ch_1", f"mom_18_ch_2"
#                 ]}

#             for condition, cond_data in dyad_data.items():
#                 for participant, part_data in cond_data.items():
#                     for channel_name in part_data.keys():
#                         col_name = f"{participant}_{group_suffix}_{channel_name}"
#                         current_val = availability[dyad_id][col_name]

#                         if current_val == "False":
#                             availability[dyad_id][col_name] = [condition]
#                         else:
#                             availability[dyad_id][col_name].append(condition)

#     df = pd.DataFrame.from_dict(availability, orient="index")
#     df.index = df.index.astype(int)
#     df = df.sort_index()

#     df.index.name = f"dyad_id ({df.shape[0]} dyads)"
#     return df



# def rsa_availability_df(rsa_dict):
#     """
#     Build availability table for RSA data.
#     Each dyad_id becomes a row.
#     Columns: participant_group_channel (e.g., infant_9_ch_0).
#     Values: list of conditions or "False".
#     """

#     availability = {}

#     for group_name, group_data in rsa_dict.items():
#         # Map group to suffix (9 or 18)
#         group_suffix = "9" if "9_months" in group_name else "18"

#         for dyad_id, dyad_data in group_data.items():
#             if dyad_id not in availability:
#                 availability[dyad_id] = {col: "False" for col in [
#                     f"infant_{group_suffix}_ch_0", f"infant_{group_suffix}_ch_1", f"infant_{group_suffix}_ch_2",
#                     f"mom_{group_suffix}_ch_0", f"mom_{group_suffix}_ch_1", f"mom_{group_suffix}_ch_2"
#                 ]}

#             for condition, cond_data in dyad_data.items():
#                 for channel_name, channel_dict in cond_data.items():
#                     df = channel_dict["data"]

#                     # Check both columns: infantRsa + motherRsa
#                     if "infantRsa" in df.columns:
#                         col_name = f"infant_{group_suffix}_{channel_name}"
#                         current_val = availability[dyad_id][col_name]
#                         if current_val == "False":
#                             availability[dyad_id][col_name] = [condition]
#                         else:
#                             availability[dyad_id][col_name].append(condition)

#                     if "motherRsa" in df.columns:
#                         col_name = f"mom_{group_suffix}_{channel_name}"
#                         current_val = availability[dyad_id][col_name]
#                         if current_val == "False":
#                             availability[dyad_id][col_name] = [condition]
#                         else:
#                             availability[dyad_id][col_name].append(condition)

#     df = pd.DataFrame.from_dict(availability, orient="index")
#     df.index = df.index.astype(int)
#     df = df.sort_index()

#     df.index.name = f"dyad_id ({df.shape[0]} dyads)"
#     return df

# def availability_view_by_participant(df, with_all_indices= False):
#     """
#     Transform the output of data_availability_df into a compact 4-column view:
#     "infant 9 month", "mom 9 month", "infant 18 month", "mom 18 month".

#     Each cell contains a string:
#         condition: channel_numbers_comma_separated
#         or "False" if no data exists
#     """
#     out_columns = ["infant 9 month", "infant 18 month", "mom 18 month", "mom 9 month"]

#     # Initialize a dict to store data per dyad
#     dyad_data_dict = {dyad: {col: {} for col in out_columns} for dyad in df.index}

#     # Step 1: Collect channels per condition
#     for dyad in df.index:
#         for col in df.columns:
#             val = df.loc[dyad, col]
#             if val == "False":
#                 continue

#             participant = col.split("_")[0]      # 'infant' or 'mom'
#             group = col.split("_")[1]           # '9' or '18'
#             ch_num = int(col.split("_")[-1])    # last part -> 0,1,2
#             age_col = f"{participant} {group} month"

#             for condition in val:
#                 dyad_data_dict[dyad][age_col].setdefault(condition, set()).add(ch_num)

#     # Step 2: Build the final string per cell
#     for dyad in dyad_data_dict:
#         for age_col in out_columns:
#             cond_dict = dyad_data_dict[dyad][age_col]
#             if cond_dict:
#                 lines = []
#                 for condition, channels in cond_dict.items():
#                     channels_str = ",".join(map(str, sorted(channels)))
#                     lines.append(f"{condition}: {channels_str}")
#                 dyad_data_dict[dyad][age_col] = "  ".join(lines)
#                 dyad_data_dict[dyad][age_col] = str(dyad_data_dict[dyad][age_col])
#             else:
#                 dyad_data_dict[dyad][age_col] = "False"

#     # Step 3: Create DataFrame
#     out_df = pd.DataFrame.from_dict(dyad_data_dict, orient="index")
#     out_df.index.name = df.index.name
#     out_df.index = out_df.index.astype(int)
#     out_df = out_df.sort_index()

#     if with_all_indices == True:
#         out_df = out_df.reindex(range(1, 91))

#     return out_df


# def availability_view_combined(ibi_df, rsa_df, with_all_indices=False, is_saving = False):
#     """
#     Merge IBI + RSA availability into a compact 4-column view:
#     "infant 9 month", "mom 9 month", "infant 18 month", "mom 18 month".

#     Each cell looks like:
#         no_toys: [ibis: 0,1,2 ; rsa:2], toys: [ibis: 0,1,2 ; rsa:1]
#     """
#     out_columns = ["infant 9 month", "mom 9 month", "infant 18 month", "mom 18 month"]

#     # Start with all dyads from both dfs
#     all_dyads = sorted(set(ibi_df.index).union(set(rsa_df.index)))
#     dyad_data_dict = {dyad: {col: {} for col in out_columns} for dyad in all_dyads}

#     # ---- Helper function to process a source df ----
#     def update_from_source(source_df, source_name):
#         for dyad in source_df.index:
#             for col in source_df.columns:
#                 val = source_df.loc[dyad, col]
#                 if val == "False":
#                     continue

#                 participant = col.split("_")[0]      # 'infant' or 'mom'
#                 group = col.split("_")[1]           # '9' or '18'
#                 ch_num = int(col.split("_")[-1])    # last part -> 0,1,2
#                 age_col = f"{participant} {group} month"

#                 for condition in val:
#                     dyad_data_dict[dyad][age_col].setdefault(condition, {"ibis": set(), "rsa": set()})
#                     dyad_data_dict[dyad][age_col][condition][source_name].add(ch_num)

#     # Fill availability
#     update_from_source(ibi_df, "ibis")
#     update_from_source(rsa_df, "rsa")

#     # ---- Build final cell strings ----
#     for dyad in dyad_data_dict:
#         for age_col in out_columns:
#             cond_dict = dyad_data_dict[dyad][age_col]
#             if cond_dict:
#                 lines = []
#                 for condition, sources in cond_dict.items():
#                     ibis_channels = ",".join(map(str, sorted(sources["ibis"]))) if sources["ibis"] else "-"
#                     rsa_channels = ",".join(map(str, sorted(sources["rsa"]))) if sources["rsa"] else "-"
#                     lines.append(f"{condition}: [ibis: {ibis_channels} ; rsa: {rsa_channels}]")
#                 dyad_data_dict[dyad][age_col] = "  ".join(lines)
#             else:
#                 dyad_data_dict[dyad][age_col] = "False"

#     # ---- Build final DataFrame ----
#     out_df = pd.DataFrame.from_dict(dyad_data_dict, orient="index")
#     out_df.index.name = "dyad_id"
#     out_df.index = out_df.index.astype(int)
#     out_df = out_df.sort_index()

#     if with_all_indices:
#         out_df = out_df.reindex(range(1, 91))
    
#     if is_saving == True:
#         with pd.ExcelWriter("Availability.xlsx") as writer:
#             out_df.to_excel(writer, sheet_name="05RSA_01IBI")

#     return out_df




# # def interleaved_availability_diff(ibis_df, peaks_df, months_first="18"):
# #     """
# #     Generate an interleaved comparison of ibis vs peaks availability DataFrames.

# #     Parameters:
# #     - ibis_df: DataFrame from availability_view_by_participant for ibis
# #     - peaks_df: DataFrame from availability_view_by_participant for peaks
# #     - months_first: "9" or "18" — which months appear first in the columns

# #     Returns:
# #     - diff_comparison: DataFrame with dyads where ibis and peaks differ,
# #       columns interleaved: ibis_X, peaks_X
# #     """
# #     # Compare the two views
# #     is_same = peaks_df == ibis_df

# #     # Find dyads where any column differs
# #     dyads_diff = is_same.index[~is_same.all(axis=1)]

# #     if len(dyads_diff) == 0:
# #         return pd.DataFrame()  # No differences

# #     # Subset only differing dyads
# #     ibis_diff = ibis_df.loc[dyads_diff].add_prefix("ibis_")
# #     peaks_diff = peaks_df.loc[dyads_diff].add_prefix("peaks_")

# #     # Define column order
# #     if months_first == "18":
# #         categories = ["infant 18 month", "mom 18 month", "infant 9 month", "mom 9 month"]
# #     else:
# #         categories = ["infant 9 month", "mom 9 month", "infant 18 month", "mom 18 month"]

# #     ordered_cols = []
# #     for cat in categories:
# #         ordered_cols.append(f"ibis_{cat}")
# #         ordered_cols.append(f"peaks_{cat}")

# #     # Concatenate side by side in desired order
# #     diff_comparison = pd.concat([ibis_diff, peaks_diff], axis=1)[ordered_cols]
# #     diff_comparison.index = diff_comparison.index.astype(int)
# #     diff_comparison = diff_comparison.sort_index()

# #     return diff_comparison




# ### RSA Data Anlysis - 9 month ###

# with open("05_rsa_calculated_raw_data.pkl", "rb") as f_raw_rsa:
#     raw_rsa_pickle = pickle.load(f_raw_rsa)

# with open("05_rsa_calculated_detrended_data.pkl", "rb") as f_detrended_rsa:
#     detrended_rsa_pickle = pickle.load(f_detrended_rsa)

# raw_rsa_data = convert_rsa_strings_to_numeric(raw_rsa_pickle)
# detrended_rsa_data = convert_rsa_strings_to_numeric(detrended_rsa_pickle)

# # rsa_data_status = data_availability_df(rsa_data)
# # rsa_availability_view = availability_view_by_participant(rsa_data_status, True)
# ibi_avail = data_availability_df(ibis_data)
# rsa_avail = rsa_availability_df(raw_rsa_data)   # same function works for rsa pickle

# combined_view = availability_view_combined(ibi_avail, rsa_avail, with_all_indices=True, is_saving = True)

# ### with toys - 40 dyads (same indices for infants and moms)
# # Raw data 
# raw_rsa_infants_9_with_toys = channel_rsa_stats_df(raw_rsa_data, participant='infantRsa', group='9_months', condition='toys', with_all_indices= True) 
# raw_rsa_moms_9_with_toys = channel_rsa_stats_df(raw_rsa_data, participant='motherRsa', group='9_months', condition='toys', with_all_indices= True) 
# # Detrended data
# detrended_rsa_infants_9_with_toys = channel_rsa_stats_df(detrended_rsa_data, participant='infantRsa', group='9_months', condition='toys', with_all_indices= True) 
# detrended_rsa_moms_9_with_toys = channel_rsa_stats_df(detrended_rsa_data, participant='motherRsa', group='9_months', condition='toys', with_all_indices= True) 

# ### without toys - 38 dyads (same indices for infants and moms)
# # Raw data 
# raw_rsa_infants_9_without_toys = channel_rsa_stats_df(raw_rsa_data, participant='infantRsa', group='9_months', condition='no_toys', with_all_indices= True) 
# raw_rsa_moms_9_without_toys = channel_rsa_stats_df(raw_rsa_data, participant='motherRsa', group='9_months', condition='no_toys', with_all_indices= True) 
# # Detrended data
# detrended_rsa_infants_9_without_toys = channel_rsa_stats_df(detrended_rsa_data, participant='infantRsa', group='9_months', condition='no_toys', with_all_indices= True) 
# detrended_rsa_moms_9_without_toys = channel_rsa_stats_df(detrended_rsa_data, participant='motherRsa', group='9_months', condition='no_toys', with_all_indices= True) 



# datasets = {
#     "raw_rsa_infants_9_with_toys": raw_rsa_infants_9_with_toys,
#     "raw_rsa_moms_9_with_toys": raw_rsa_moms_9_with_toys,
#     "raw_rsa_infants_9_without_toys": raw_rsa_infants_9_without_toys,
#     "raw_rsa_moms_9_without_toys": raw_rsa_moms_9_without_toys,
#     "detrended_rsa_infants_9_with_toys": detrended_rsa_infants_9_with_toys,
#     "detrended_rsa_moms_9_with_toys": detrended_rsa_moms_9_with_toys,
#     "detrended_rsa_infants_9_without_toys": detrended_rsa_infants_9_without_toys,
#     "detrended_rsa_moms_9_without_toys": detrended_rsa_moms_9_without_toys,

# }

# with pd.ExcelWriter("05_rsa_9_month.xlsx") as writer:

#     for name, df in datasets.items():
#         # clean sheet name if too long
#         sheet_name = name[:31]  # Excel limit for sheet names
#         df.to_excel(writer, sheet_name=sheet_name)
