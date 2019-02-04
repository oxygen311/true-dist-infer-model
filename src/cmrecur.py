import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def cm(m, t, x):
    return x ** (m - 1) * t ** ((t + 1) * m) / (x + t) ** (m * t + 2 * m - 2) * math.gamma(
        m * t + 2 * m - 2) / math.gamma(m * t + m) / math.gamma(m + 1)


def cm_nong(m, t, x):
    if m == 1:
        return t ** (t + 1) / (x + t) ** t
    else:
        prev = cm_nong(m - 1, t, x)
        return prev * x * t ** (t + 1) / (x + t) ** (t + 2)


def cm2(m, t, x):
    return cm_nong(m, t, x) * math.gamma(m * t + 2 * m - 2) / math.gamma(m * t + m) / math.gamma(m + 1)


def cm3(m, t, x):
    pr = 1
    for j in range(m - 2):
        pr *= (m * t + m + j) / (j + 2)
    return cm_nong(m, t, x) * pr / m


cache = defaultdict(lambda: defaultdict(lambda: {}))
def cm4(m, t):
    def f(x):
        if m in cache and t in cache[m] and x in cache[m][t]:
            return cache[m][t][x]
        if m == 1:
            return t ** t / (x + t) ** t
        else:
            prev = cm4(m - 1, t)(x)
            pr = 1
            for j in range(m - 2):
                pr *= (m * t + m + j) / (m * t - t + m - 1 + j)
            pr *= ((m - 1) * t + 2 * m - 4) / m
            ret = prev * x * t ** (t + 1) / (x + t) ** (t + 2) * pr
            if not (m in cache and t in cache[m] and x in cache[m][t]):
                cache[m][t][x] = ret
            return ret

    return f


b_gamma = lambda t: lambda x: 1 - cm4(1, t)(x)
d_gamma = lambda t, mx: lambda x: 1 - np.sum([cm4(m, t)(x) for m in range(1, mx)])
d_gamma_b = lambda t, mx: lambda x: d_gamma(t, mx)(x) / b_gamma(t)(x)
#
# args = []
# cms = []
# cms2 = []
# for i in range(150):
#     args.append(i / 100)
#     # cms.append(cm(80,0.3,i/100))
#     cms2.append(d_gamma(0.3, 50)(i / 100))
#
# # plt.plot(args,cms)
# plt.plot(args, cms2)
# plt.show()
