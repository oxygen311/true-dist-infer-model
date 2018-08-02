from src.dirichlet_graph import DirichletDnkGraph
from src.classic_graph import DnkGraph
import numpy as np
import json
from datetime import datetime
from src.estimators import get_d_from_cycles
import random
import multiprocessing as mp
from collections import defaultdict

X_MIN = 0
X_MAX = 3
X_STEP = 0.01
xs = np.arange(X_MIN, X_MAX, X_STEP)
x = 0.75


def run(tries, workers):
    ns = defaultdict(lambda: [])

    for i in range(int(tries / workers)):
        if i != 0:
            print(i, "of", int(tries / workers), ";", round(i / tries * workers * 100, 1), "%")
        for n in range(100, 501, 50):
            g = DirichletDnkGraph(n)
            breaks = 0

            while breaks <= int(round(x * n / 2)):
                g.do_k_break()
                breaks += 1
            ns[n].append([n, breaks, dict(g.count_cycles())])
    return dict(ns)


def measure_c(tries, workers):
    pool = mp.Pool()
    results = [pool.apply_async(run, [tries, workers]) for _ in range(workers)]

    res = defaultdict(lambda: [])
    for r in results:
        for k, v in r.get().items():
            res[k] += v

    return dict(res)


if __name__ == "__main__":
    TRIES = 1000
    WORKERS = 8

    measured_c = measure_c(TRIES, WORKERS)
    open("data/diff_ns/dirichlet_%d.txt" % (TRIES), 'w').write(json.dumps(measured_c))
