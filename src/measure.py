from src.dirichlet_graph import DirichletDnkGraph
from src.classic_graph import DnkGraph
import numpy as np
import json
from datetime import datetime
from src.estimators import get_d_from_cycles
import random
import multiprocessing as mp

X_MIN = 0
X_MAX = 3
X_STEP = 0.01
xs = np.arange(X_MIN, X_MAX, X_STEP)


def run(n, tries, workers):
    ys = [[] for _ in xs]
    for i in range(int(tries / workers)):
        print(i, "of", int(tries / workers), "N =", n, datetime.now())
        g = DnkGraph(n)
        breaks = 0

        for x_index, x in enumerate(xs):
            while breaks <= int(round(x * n / 2)):
                g.do_k_break()
                g.check_correctness()
                breaks += 1
            ys[x_index].append([n, breaks, dict(g.count_cycles())])
    return ys


def measure_c(n, tries, workers):
    pool = mp.Pool()
    results = [pool.apply_async(run, [n, tries, workers]) for _ in range(workers)]

    res = [[] for _ in xs]
    for r in results:
        for i, x in enumerate(r.get()):
            res[i] += x

    return res


if __name__ == "__main__":
    TRIES = 100
    WORKERS = 8

    for n in range(100, 500, 100):
        if n % 50 == 0:
            print("Skip:", n)
            continue
        measured_c = measure_c(n, TRIES, WORKERS)
        print(len(measured_c))
        open("data/diff_ns/dirichlet_%d_%d.txt" % (n, TRIES), 'w').write(json.dumps(measured_c))
