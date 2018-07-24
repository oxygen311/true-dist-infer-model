import matplotlib.pyplot as plt
import math
from itertools import repeat, filterfalse, dropwhile
from sympy.ntheory import factorint
import random
import numpy as np


def graf():
    plt.grid(True)
    plt.xlabel("number")
    plt.ylabel("factors")
    N = 100000

    num_of_factors = [0] * N
    cur_factors_number = 0
    for i in range(1, N):
        cur_factors_number += 1 if len(factorint(i)) == 1 else 0
        num_of_factors[i] = cur_factors_number
    xs = range(2, N)

    plt.plot(xs, list(map(lambda n: 1.1 * n / math.log(n), xs)), 'k-', label='by formula')
    plt.plot(xs, list(map(lambda n: num_of_factors[n], xs)), 'r-', label='by experiment')

    plt.legend(loc=1)

    plt.show()

    # my_list = [x for x in my_list if x.attribute == value]

def c2_rec(n, k):
    def a(l2, m2):
        return (l2 + 1) * (m2 + 1) * (n + l2 + m2 + 2) * (n + l2 + m2 + 3) + (2 - 2 * k) * (l2 + 2) * (m2 + 1) * (
                n + l2 + m2 + 3) + (2 - 2 * k) * (l2 + 1) * (m2 + 2) * (n + l2 + m2 + 3) + (2 - 2 * k) ** 2 * (
                       l2 + 2) * (m2 + 2)

    def gamma(l2, m2):
        if l2 <= 0 and m2 <= 0:
            return a(0, 0) / (n * (n + 1) * (n + 2) * (n + 3))

        if l2 > 0:
            return gamma(l2 - 2, m2) * 4 * (k - 1) ** 2 * a(l2, m2) / \
                   (a(l2 - 2, m2) * (n + l2 + m2 + 2) * (n + l2 + m2 + 3))

        return gamma(l2, m2 - 2) * 4 * (k - 1) ** 2 * a(0, m2) / \
               (a(0, m2 - 2) * (n + m2 + 2) * (n + m2 + 5))

    return k * n * n * sum([sum([gamma(2 * l, 2 * m) for m in range(50)]) for l in range(50)])


def c1_rec(n, k):
    def beta(l2):
        if l2 <= 0:
            return (n - 2 * k) / n

        return beta(l2 - 2) * 4 * k ** 2 * (n + l2 - 2 * k) / \
               ((n + l2 - 1) * (n + l2) * (n + l2 - 2 - 2 * k))

    betas = [beta(0)]
    l2 = 0
    while abs(betas[-1]) > 1e-4 and l2 < 50:
        l2 += 2
        betas.append(beta(l2))
    return n * sum(betas)

# graf()
# a = filterfalse(lambda x: x > 10, repeat(random.randint(1, 20))).__next__()
# a = any(x > 10 for x in repeat(12))

def generate_new_permutation(n):
    def is_correct(p):
        return not any(
            (p[i] % 2 == 0 and p[i] == p[i + 1] - 1) or (p[i + 1] % 2 == 0 and p[i + 1] == p[i] - 1)
            for i in range(0, len(p), 2)
        )

    return next(filter(lambda p: is_correct(p), repeat(np.random.permutation(n))))


print(generate_new_permutation(4))

# a = dropwhile(lambda x: x < 5, repeat(random.randint(1, 10)))

# print(next(a))

# for x in a:
#     print(x)

