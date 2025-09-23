import pandas as pd
from config import folder_paths, Config_01_ibi_after_extraction
from def_process_csv_folder import process_csv_folder


ibi_after_extraction_folder = folder_paths["01_ibi_after_extraction"]


process_csv_folder(ibi_after_extraction_folder, Config_01_ibi_after_extraction, "01_ibi_after_extraction_data", True)

