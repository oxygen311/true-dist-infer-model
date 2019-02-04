from scipy import optimize
from scipy.special import gamma
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.drawer import d_over_n_dir
from src.cmrecur import cm4, b_gamma, d_gamma


pars_indicator = lambda t: lambda x: x / 2 - d_gamma(t, 150)(x) - 1e-15
b_pars_indicator = lambda t: lambda x: abs(x / 2 - d_gamma(t, 150)(x)) > 1e-4
# pars_dir_indicator = lambda x: x / 2 - d_over_n_dir(x) - 1e-15

find_parsimony_bound = lambda t: optimize.bisect(pars_indicator(t), 1e-6, 3, xtol=1e-5)
# find_dir_parsimony_bound = lambda x: optimize.bisect(pars_dir_indicator, 1e-6, 3, xtol=1e-4)

d_over_b = lambda t: lambda x: d_gamma(t, 50)(x) / b_gamma(t)(x)


def boolean_binary_search(f):
    l, r, m = 0.0, 3.0, 0.0
    while r - l > 1e-3:
        m = (r + l) / 2
        # print(m, f(m))
        if f(m):
            r = m
        else:
            l = m
    return m


def main():
    start_time = time.time()
    sns.set(style="whitegrid")
    ts = np.arange(0.01, 2, 0.01)

    # for x in np.arange(0.1, 2, 0.1):
    #     print(x, )
    # vfunc = np.vectorize(find_parsimony_bound)
    print(pars_indicator(1)(0.1))
    print(pars_indicator(1)(0.5))
    print(pars_indicator(1)(0.51))
    print(pars_indicator(1)(0.7))
    print(pars_indicator(1)(0.8))
    print(boolean_binary_search(b_pars_indicator(1)))

    dbs = []
    xs = []
    for t in ts:
        x = find_parsimony_bound(t)
        x = boolean_binary_search(b_pars_indicator(t))
        db = d_over_b(t)(x)
        dbs.append(db)
        xs.append(x)
    plt.plot(ts, xs)
    plt.xlabel("t")
    plt.ylabel("x")

    plt.savefig('pars.pdf', transparent=True)
    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))


main()
