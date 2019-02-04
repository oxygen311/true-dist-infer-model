from itertools import combinations
from src.hic.blocks_parser import parse_to_df, create_bg
from src.hic.real_data_full_df import sp_blocks_from_df, uniq_predicate, species, df_file
from collections import defaultdict

apcf = "EUA"
res = "300Kbp"

df = parse_to_df(df_file % (apcf, res))
dct = defaultdict(lambda: 0)

for sp1, sp2 in combinations(species, 2):
    allowed_blocks = df['block'].unique().tolist()
    allowed_blocks = list(filter(uniq_predicate((sp_blocks_from_df(df, sp1))), allowed_blocks))
    allowed_blocks = list(filter(uniq_predicate((sp_blocks_from_df(df, sp2))), allowed_blocks))

    g = create_bg(df, sp1, sp2, allowed_blocks)
    print(sp1, sp2, g.c())

    for k, v in g.non_trivial_blocks().items():
        dct[k] += v

for b in df['block'].unique().tolist():
    if dct[b] == 0:
        print(b, dct[b])

print("count:", sum(1 if v == 0 else 0 for v in dct.values()), "of", len(df['block'].unique().tolist()))