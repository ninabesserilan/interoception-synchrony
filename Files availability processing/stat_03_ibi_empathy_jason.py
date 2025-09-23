
from defs_empathy_availability import build_empathy_dict
from config import folder_paths, Config_03_jason_ibi_empathy_data

folder_9mo = folder_paths['03_ibi_empathy_data_9_month']
folder_18mo = folder_paths['03_ibi_empathy_data_18_month']

empathy_dic_18, df_18 = build_empathy_dict(folder_18mo, '18', Config_03_jason_ibi_empathy_data, with_all_indices= True, is_saving = True)

empathy_dic_9, df_9 = build_empathy_dict(folder_9mo, '9', Config_03_jason_ibi_empathy_data, with_all_indices= True, is_saving = True)



