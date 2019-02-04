import pandas as pd
from src.hi_c_parser import HiCData
import numpy as np

df = pd.read_pickle("data/df.pkl")
sp1 = "hg18"
sp2 = "mm8"
chr_sp1 = 1
chr_sp2 = 4

df_filtered_sp1 = df.loc[(df['species'] == sp1) & (df['chr'] == str(chr_sp1))].sort_values(by=['chr_beg'])
df_filtered_sp2 = df.loc[(df['species'] == sp2) & (df['chr'] == str(chr_sp2))]
prev_row_sp2 = None

hi_c = HiCData(250000)
print("Mean in all data:", hi_c.get_mean_v())

vs = []

for i, row_sp1 in df_filtered_sp1.iterrows():
    tmp_df = df_filtered_sp2.loc[df_filtered_sp2['block'] == row_sp1['block']]
    # check if block in exist in species #2
    if tmp_df.empty:
        continue

    row_sp2 = tmp_df.iloc[0]
    # check if block is going after previous in species #2
    if prev_row_sp2 is not None:
        s = {prev_row_sp2['chr_beg'], prev_row_sp2['chr_end'], row_sp2['chr_beg'], row_sp2['chr_end']}
        if len(s) < 4:
            continue
        vs.append(hi_c.get_nearest_value(row_sp2['chr_beg'], prev_row_sp2['chr_beg']))
        vs.append(hi_c.get_nearest_value(row_sp2['chr_beg'], prev_row_sp2['chr_end']))
        vs.append(hi_c.get_nearest_value(row_sp2['chr_end'], prev_row_sp2['chr_beg']))
        vs.append(hi_c.get_nearest_value(row_sp2['chr_end'], prev_row_sp2['chr_end']))
        # vs.append(hi_c.get_nearest_value(row_sp2['chr_beg'], row_sp2['chr_end']))

    prev_row_sp2 = row_sp2

print("Mean in interesting locations:", np.mean(vs))