import pandas as pd

df = pd.read_pickle("data/df.pkl")
sp1 = "hg18"
sp2 = "mm8"

with open('%s-%s.tsv' % (sp1, sp2), 'w') as f:
    for block in df['block'].unique():
        in_sp1 = df.loc[(df['species'] == sp1) & (df['block'] == block)]
        in_sp2 = df.loc[(df['species'] == sp2) & (df['block'] == block)]
        print("CL-%s\tchr%s\t%s\t%s\t%s\tCL-%s\tchr%s\t%s\t%s\t%s"
              % (block,
                 in_sp1['chr'].values[0],
                 in_sp1['chr_beg'].values[0],
                 in_sp1['chr_end'].values[0],
                 "1" if in_sp1['orientation'].values[
                            0] == "+" else "-1",
                 block,
                 in_sp2['chr'].values[0],
                 in_sp2['chr_beg'].values[0],
                 in_sp2['chr_end'].values[0],
                 "1" if in_sp2['orientation'].values[
                            0] == "+" else "-1",
                 ), file=f)
        print(block)