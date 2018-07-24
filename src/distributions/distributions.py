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
    # return sorted(xs / np.sum(xs), reverse=True)


def generate_with_line(n):
    rs = [random.random() for _ in range(n - 1)]

    line = [0]
    for x in sorted(rs):
        line.append(x)
    line.append(1)

    return [line[i + 1] - line[i] for i in range(n)]


def generate_partitioning_process(n, steps_func=lambda n: n ** 2):
    xs = [1] * n

    for _ in range(steps_func(n)):
        # print(xs / np.sum(xs))
        inds = np.random.choice(n, 2, replace=False, p=list(xs / np.sum(xs)))
        sum = xs[inds[0]] + xs[inds[1]]
        rnd = random.random()

        xs[inds[0]] = rnd * sum
        xs[inds[1]] = (1 - rnd) * sum

    return xs / np.sum(xs)


def generate_partitioning_process_2rnds(n, steps_func=lambda n: n ** 2):
    xs = [1] * n
    steps = steps_func(n)

    for i in range(steps):
        if i != 0:
            print(i / steps * 100, "%")
        # print(xs / np.sum(xs))
        inds = np.random.choice(n, 2, replace=False, p=list(xs / np.sum(xs)))

        sum = xs[inds[0]] + xs[inds[1]]
        rnd1 = random.random() * xs[inds[0]]
        rnd2 = random.random() * xs[inds[1]]

        xs[inds[0]] = rnd1 + rnd2
        xs[inds[1]] = sum - (rnd1 + rnd2)

    return xs / np.sum(xs)


def generate_random_matrix_mult(n, steps_func=lambda n: n ** 2, k=2):
    def generate_new_weights(ws):
        r = np.array([generate_normed_exp(len(ws)) for _ in range(len(ws))])
        return np.dot(r.transpose(), ws)

    p = [1] * n
    steps = steps_func(n)

    for i in range(steps):
        print(i / steps * 100, "%")

        inds = np.random.choice(n, k, replace=False, p=list(p / np.sum(p)))
        old_ws = list(itemgetter(*inds)(p))

        new_ws = generate_new_weights(old_ws)
        for ind, new_w in zip(inds, new_ws):
            p[ind] = new_w

    return p / np.sum(p)


def generate_tao(n):
    us = [random.random() for _ in range(n)]

    # def enum(xs):
    #     print(xs)
    #     if xs.size == 0:
    #         return []
    #
    #     ret = xs[0]
    #     return [ret] + enum(xs[1:] / (1 - ret))
    # sigma = enum(np.array(us))

    factors = [1]
    sigma = [us[0]]
    for i in range(1, n):
        factors.append(factors[i - 1] * (1 - us[i]))
        sigma.append(us[i] * factors[i])

    return sigma / np.sum(sigma)


def generate_with_factorization(n):
    rnd = randint(int(math.e ** n), int(2 * math.e ** n))
    log_rnd = math.log(rnd)

    xs = []
    factorization = factorint(rnd)

    for fact in factorization:
        for i in range(factorization[fact]):
            xs.append(math.log(fact) / log_rnd)

    print("rnd =", rnd, "\nlen_fact =", len(xs), "\nfactorization =", factorization)

    return list(reversed(xs))


def compare_two_distributions(f1, f2):
    NUMBER_OF_BASKETS = 100
    N = 100
    CHECK_N = 10
    TESTS = 100
    ERROR_FUNC = lambda x, y: abs(x - y)

    def get_baskets_from_cut(cut):
        baskets = [0] * NUMBER_OF_BASKETS
        for x in cut:
            baskets[int(x * NUMBER_OF_BASKETS)] += 1
        return baskets

    def get_cut(measured, i):
        return list(map(lambda x: x[i], measured))

    def diff_between_baskets(b1, b2):
        return list(ERROR_FUNC(x, y) for x, y in zip(b1, b2))

    measured1 = [sorted(f1(N), reverse=True) for _ in range(TESTS)]
    measured2 = [sorted(f2(N), reverse=True) for _ in range(TESTS)]

    some_number = 0
    for i in range(CHECK_N):
        cut1 = get_cut(measured1, i)
        bas1 = get_baskets_from_cut(cut1)

        cut2 = get_cut(measured2, i)
        bas2 = get_baskets_from_cut(cut2)

        some_number += sum(diff_between_baskets(bas1, bas2))

    print(some_number)
    return some_number


def compare_with_random_length_dist(f_random_len, f_fixed_len):
    NUMBER_OF_BASKETS = 50
    N = 20
    CHECK_N = 2
    TESTS = 500
    ERROR_FUNC = lambda x, y: abs(x - y)

    def get_baskets_from_cut(cut):
        baskets = [0] * (NUMBER_OF_BASKETS + 1)
        for x in cut:
            baskets[int(x * NUMBER_OF_BASKETS)] += 1
        return baskets

    def get_cut(measured, i):
        return list(map(lambda x: x[i], measured))

    def diff_between_baskets(b1, b2):
        return list(ERROR_FUNC(x, y) for x, y in zip(b1, b2))

    measured1 = []
    measured2 = []
    for _ in range(TESTS):
        tmp_dist = f_random_len(N)
        if len(tmp_dist) == 1:
            break
        measured1.append(tmp_dist)
        measured2.append(sorted(f_fixed_len(len(measured1[-1])), reverse=True))

    some_number = 0
    for i in range(CHECK_N):
        cut1 = get_cut(measured1, i)
        bas1 = get_baskets_from_cut(cut1)

        cut2 = get_cut(measured2, i)
        bas2 = get_baskets_from_cut(cut2)

        some_number += sum(diff_between_baskets(bas1, bas2))

    return some_number


if __name__ == "__main__":
    n = 20
    d1 = generate_normed_exp(n)
    d2 = generate_random_matrix_mult(n, lambda n: n ** 3, 6)
    print(ks_2samp(d1, d2))
