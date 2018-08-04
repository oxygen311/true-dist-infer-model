import matplotlib.pyplot as plt
import math
import numpy as np
import json
from src.estimators import TannierEstimator, DirichletEstimator, DataEstimator, UniformEstimator, FirstCmsDirEstimator
from time import time
from scipy.special import hyp2f1
from src.drawer import est_error
import seaborn as sns
import pandas as pd
import multiprocessing as mp

# plotly
file_template = "data/diff_ns/dirichlet_%d.txt"


def run(ds, n):
    data_func = est_error(FirstCmsDirEstimator(5))
    actual_func = lambda p: data_func(*p)
    res = []
    for i, d in enumerate(ds):
        print(i, "of", len(ds),  "n =", n)
        error = actual_func(d)
        res.append((n, error))
    return res


if __name__ == "__main__":
    sns.set(style="whitegrid", font="serif")
    columns = ["n", "method", "relative error"]
    workers = 4
    tries = 1000
    data = json.loads(open(file_template % (tries), 'r').read())

    data_function = est_error(FirstCmsDirEstimator(5))
    actual_func = lambda p: data_function(*p)
    df = pd.DataFrame(columns=columns)
    for n, arr in data.items():
        print("cur n:", n)

        split = np.array_split(arr, workers)
        pool = mp.Pool()
        results = [pool.apply_async(run, [sp, str(n)]) for sp in split]

        for r in results:
            for n, e in r.get():
                df = df.append({'n': n, 'method': 'our method', 'relative error': e}, ignore_index=True)
                # df = df.append({'n': n, 'method': 'our', 'relative error': e + 0.01}, ignore_index=True)
    # df.to_pickle("data/diff_n_errors_our_100_500(1000)2.pkl")
    # df2 = pd.read_pickle("data/diff_n_errors_tan_100_500.pkl")
    # print(df)
    # print(df2)
    # df = df.append(df2, ignore_index=True)

    print(df)
    df["absolute value of relative error"] = df["relative error"].apply(lambda x: abs(x))

    s = sns.boxplot(x="n", y="absolute value of relative error", hue="method", data=df, whis=[5, 95], showfliers=False,
                    palette="RdBu")

    plt.savefig('ns.pdf')
    plt.legend(frameon=True)

    plt.show()
