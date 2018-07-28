import matplotlib.pyplot as plt
import math
import numpy as np
import json
from src.estimators import TannierEstimator, DirichletEstimator, DataEstimator, UniformEstimator, FirstCmsDirEstimator
from time import time
from scipy.special import hyp2f1
from src.drawer import est_error

# plotly
file_template = "data/diff_ns/dirichlet_%d_%d.txt"

if __name__ == "__main__":
    ns = range(50, 801, 50)
    tries = 1000
    xs = np.arange(0.25, 1.50, 0.25)
    ys = [[] for _ in xs]

    data_function = est_error(FirstCmsDirEstimator(5))
    actual_func = lambda p: data_function(*p)

    for n in ns:
        data = json.loads(open(file_template % (n, tries), 'r').read())
        print("cur n:", n)
        for i, x in enumerate(xs):
            ds = data[int(x * 100)]
            tmp = list(filter(lambda el: el, [abs(actual_func(d)) for d in ds])) or [0]
            ys[i].append(np.mean(tmp))

    for i, x in enumerate(xs):
        plt.plot(ns, ys[i], label=("x = " + str(x)))

    plt.legend(loc=1)
    plt.grid(True)
    plt.xlabel("n")
    plt.ylabel("mean relative error abs")

    plt.savefig('ns.png')
    plt.show()