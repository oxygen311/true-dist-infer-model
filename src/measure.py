from src.dirichlet_graph import DirichletDnkGraph
from src.classic_graph import DnkGraph
import numpy as np
import json
from datetime import datetime
from src.estimators import get_d_from_cycles
import random

N_MIN = 2000
N_MAX = 4000

TRIES = 200

X_MIN = 0
X_MAX = 3
X_STEP = 0.01


def measure_c(xs):
    ys = [[] for _ in xs]

    for i in range(TRIES):
        N = random.randint(N_MIN, N_MAX)
        print(i, "of", TRIES, "N =", N, datetime.now())
        g = DnkGraph(N)
        breaks = 0

        for x_index, x in enumerate(xs):
            cycle_breaks = 0
            while breaks <= int(round(x * N / 2)):
                before_break_d = get_d_from_cycles(g.count_cycles())
                g.do_k_break()
                g.check_correctness()
                after_break_d = get_d_from_cycles(g.count_cycles())
                breaks += 1
                if after_break_d <= before_break_d:
                    cycle_breaks += 1
            if cycle_breaks != 0:
                print(x, cycle_breaks)
            ys[x_index].append([N, breaks, cycle_breaks])
    return ys


if __name__ == "__main__":
    xs = np.arange(X_MIN, X_MAX, X_STEP)
    # DIRICHLET_INPUT_FILE = "dirichlet_2000.txt"
    # dir = json.loads(open(DIRICHLET_INPUT_FILE, 'r').read())

    measured_c = measure_c(xs)
    # open("data/classic_data_random_round_" + str(TRIES) + ".txt", 'w').write(json.dumps(measured_c))
    open("data/dirichlet_breakdown_of_a_cycle" + str(TRIES) + ".txt", 'w').write(json.dumps(measured_c))
