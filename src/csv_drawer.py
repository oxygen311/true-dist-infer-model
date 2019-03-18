import pandas as pd
import matplotlib.pyplot as plt
from src.drawer import d_over_b_dir

folder = "sim_data2/"


def draw_file(file, label):
    df = pd.read_csv(folder + file)

    df['x'] = df.apply(lambda row: round(row['k'] * 2 / row['n'], 2), axis=1)
    df['d_b'] = df.apply(lambda row: row['d'] / row['b'], axis=1)

    xs = df['x'].unique().tolist()
    d_bs = list([df.loc[df['x'] == x]['d_b'].mean() - d_over_b_dir(x) for x in xs])

    plt.plot(xs, d_bs, label=label)


draw_file("dir_dist_n1000_runs100_chr1.csv", "1")
draw_file("dir_dist_n1000_runs100_chr10.csv", "10")
draw_file("dir_dist_n1000_runs100_chr20.csv", "20")
plt.grid(True)
plt.legend(loc=1)
plt.show()