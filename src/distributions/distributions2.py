import random
import numpy as np
import pandas as pd
import math
from random import randint
from sympy.ntheory import factorint
from scipy.stats import ks_2samp
from operator import itemgetter


def generate_normed_exp(n):
    xs = [random.expovariate(1) for _ in range(n)]
    return xs / np.sum(xs)


def generate_random_matrix_mult(n, steps_func=lambda n: n ** 2, k=2):
    def generate_new_weights(ws):
        if len(ws) == 2:
            r1 = random.random()
            r2 = random.random()
            return [ws[0] * r1 + ws[1] * r2, (1 - r1) * ws[0] + (1 - r2) * ws[1]]

        r = np.array([generate_normed_exp(len(ws)) for _ in range(len(ws))])
        return np.dot(r.transpose(), ws)

    p = [1] * n
    steps = steps_func(n)

    print("before", np.sum(p))
    for i in range(steps):
        print(i / steps * 100, "%")

        inds = np.random.choice(n, k, replace=False, p=list(p / np.sum(p)))
        new_ws = generate_new_weights(itemgetter(*inds)(p))

        for ind, new_w in zip(inds, new_ws):
            p[ind] = new_w

    print("after", np.sum(p))


    return p / np.sum(p)


if __name__ == "__main__":
    n = 100

    d1 = generate_normed_exp(n)
    d2 = generate_random_matrix_mult(n, lambda n: n ** 2, 2)

    print(ks_2samp(d1, d2))
