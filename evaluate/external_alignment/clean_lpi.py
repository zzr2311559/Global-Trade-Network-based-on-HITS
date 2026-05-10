import pandas as pd
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

input_csv_path = os.path.join(BASE_DIR, "data", "WB_LPI.csv")
output_dir = os.path.join(BASE_DIR, "evaluate", "external_alignment", "External_Metrics")
output_csv_path = os.path.join(output_dir, "europe_lpi_2021.csv")

os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(input_csv_path):
    print(f"cannot find {input_csv_path}")
    sys.exit(1)

print(f"Reading: {input_csv_path} ...")
raw_df = pd.read_csv(input_csv_path, low_memory=False)

# - TIME_PERIOD == 2023: 注意，没有2021的数据
# - INDICATOR == 'LPI_OVR'
filtered_df = raw_df[
    (raw_df['TIME_PERIOD'] == 2023) & 
    (raw_df['INDICATOR'] == 'LPI_OVR')
]

filtered_df = filtered_df[~filtered_df['COMP_BREAKDOWN_2_LABEL'].str.contains('Rank|bound', case=False, na=False)]

final_df = filtered_df[['REF_AREA_LABEL', 'OBS_VALUE']].copy()
final_df.columns = ['Country', 'LPI_Score']

final_df = final_df.drop_duplicates(subset=['Country'])
final_df.to_csv(output_csv_path, index=False)

print(f"Data cleaned for {len(final_df)} countries")
print(f"Result saved to {output_csv_path}")