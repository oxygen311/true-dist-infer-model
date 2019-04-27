import pandas as pd
import numpy as np
from src.estimators import MoretEstimator
import matplotlib.pyplot as plt


input_file = "output/corrected_estimations.csv"
input_range_file = "output/range_estimations_2_5_97_5.csv"

def array_stats(name, es):
    h = plt.hist(es, bins=20, alpha=0.2, edgecolor='k', cumulative=False, normed=False, label="hist")
    print(name)
    print(np.min(es), np.percentile(es, 25), np.mean(es), np.percentile(es, 75), np.max(es))
    print()
    plt.savefig(name + ".pdf")
    plt.show()

sp_dist = {
    "rheMac3" : "Rhesus",
    "galGal4" : "Chicken",
    "EUT" : "Eutheria",
    "capHir1" : "Goat",
    "monDom5" : "Opossum ",
    "hg19" : "Human",
    "mm10" : "Mouse",
    "susScr3" : "Pig",
    "EUA" : "Euarchontoglires"
}

df = pd.read_csv(input_file)
df_range = pd.read_csv(input_range_file)
print(df.columns.values, end="\n\n")
print(df_range.columns.values, end="\n\n")

df['sum of d'] = df.apply(lambda row: row['d: s1 - apcf'] + row['d: s2 - apcf'], axis=1)

df['error of d'] = df.apply(
    lambda row: abs(row['sum of d'] - row['d: s1 - s2']) / row['sum of d'], axis=1)

df['error of gamma (of d)'] = df.apply(
    lambda row: abs(row['sum of d'] - row['gamma(1/3): s1 - s2']) / row['sum of d'], axis=1)

df['error of gamma (of gamma)'] = df.apply(
    lambda row: abs(row['gamma(1/3): s1 - apcf'] + row['gamma(1/3): s2 - apcf'] - row['gamma(1/3): s1 - s2'])
                / (row['gamma(1/3): s1 - apcf'] + row['gamma(1/3): s2 - apcf']), axis=1)

df['error of dir (of d)'] = df.apply(
    lambda row: abs(row['sum of d'] - row['dir: s1 - s2']) / row['sum of d'], axis=1)

df['error of unif (of d)'] = df.apply(
    lambda row: abs(row['sum of d'] - row['unif: s1 - s2']) / row['sum of d'], axis=1)

df.sort_values('error of gamma (of d)', inplace=True)
print_rows_count = 10

df_test = df.sort_values('error of gamma (of gamma)')

# print(df['error of d'].mean())
error_of_d = []
error_gamma_of_d = []
error_gamma_of_gamma = []
for (index, row) in df_test.iterrows():
    if not row['parsimony']:
        print(row['error of gamma (of gamma)'] * 100, "\t\t", row['species1'], row['species2'], row['apcf'])
        error_of_d.append(row['error of d'] * 100)
        error_gamma_of_d.append(row['error of gamma (of d)'] * 100)
        error_gamma_of_gamma.append(row['error of gamma (of gamma)'] * 100)
    else:
        print("    <PARSIMONY>   ", row['species1'], row['species2'], row['apcf'])

print()
array_stats("error_of_d", error_of_d)
# array_stats("error_gamma_of_d", error_gamma_of_d)
# array_stats("error_gamma_of_gamma", error_gamma_of_gamma)

moret_est = MoretEstimator()

for (index, row), i in zip(df.iterrows(), range(print_rows_count)):
    # print(row['error of gamma (of d)'])
    if i == 2 or i == 3 or i == 1 or (i > 5 and i < 8):
        continue

    range_row = df_range.loc[(df_range['species1'] == row['species1'])
                             & (df_range['species2'] == row['species2'])
                             & (df_range['apcf'] == row['apcf'])]

    moret_k = moret_est.predict_k_by_bn(row['b: s1 - s2'], row['n'])

    print("\makecell{" +
            "\emph{" + f"{sp_dist[row['species1']]}" + "}" +
            "---"
            "\emph{" + f"{sp_dist[row['species2']]}" + "}" + "}",

            "\emph{" + f"{sp_dist[row['apcf']]}" + "}",
          f"{int(row['sum of d'])}",

          "\makecell{" + f"{int(row['d: s1 - s2'])} \\\\" +
          ('%.2f' % (round(row['error of d'], 4) * 100)) + " \\% }\n",

          "\makecell{" + f"{int(row['unif: s1 - s2'])} \\\\ " + (
                      '%.2f' % (round(row['error of unif (of d)'], 4) * 100)) + " \\% }",

          "\makecell{" + f"{int(row['dir: s1 - s2'])} " +
          f"({range_row['dir_min: s1 - s2'].values[0]} --- {range_row['dir_max: s1 - s2'].values[0]}) \\\\" +
          ('%.2f' % (round(row['error of dir (of d)'], 4) * 100)) + " \\% }",


          "\makecell{" + f"{int(row['gamma(1/3): s1 - s2'])} " +
          f"({range_row['gamma0.3_min: s1 - s2'].values[0]} --- {range_row['gamma0.3_max: s1 - s2'].values[0]}) \\\\" +
          ('%.2f' % (round(row['error of gamma (of d)'], 4) * 100)) + " \\% }",

          # "\makecell{" + f"{moret_k}" + " \\\\ " + ('%.2f' % (100 * (row['sum of d'] - moret_k) / row['sum of d'])) + "}",

          sep=" & ", end="")

    print(" \\\\ \\midrule")

df1 = df[['species1', 'species2', 'error of d', 'error of gamma (of d)']]
df1.to_csv("nikita_errors.csv", index=False)
print(df1)



    # if index > 10:
    #     break
