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

    return [x for r in results for x in r.get()]


if __name__ == "__main__":
    TRIES = 1000
    WORKERS = 8

    for n in range(350, 2500, 50):
        measured_c = measure_c(n, TRIES, WORKERS)
        open("data/diff_ns/dirichlet_%d_%d.txt" % (n, TRIES), 'w').write(json.dumps(measured_c))
