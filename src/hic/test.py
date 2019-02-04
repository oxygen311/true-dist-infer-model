import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import scipy
import scipy.stats
import seaborn as sns

# file = "/Users/alexey/Downloads/mES/eij/eij.chr1"
#
# data = np.loadtxt(file)
# data = np.random.randn(10, 12)

# f = lambda x: 0 if x == 0 else math.log(x, math.e)
# g = lambda x: 0 if x == 0 else 1
# data_f = np.vectorize(f)
# data_g = np.vectorize(g)
# print(len(data))
#
# # for i in range(len(data)):
# #     for j in range(len(data)):
# #         if abs(i - j) < 60:
# #             data[i][j] = 0
#
# not_zeros = data_g(data).sum()
# print(data.mean())
# print(data.sum() / not_zeros)
# print(data.min())
# print(data.max(), data.argmax() // len(data), data.argmax() % len(data))
# print(np.percentile(data, 75))
# print(np.percentile(data, 90))
# print(np.percentile(data, 99))
#
# # data = data_f(data)
#
# print(data.max())
# print(data.min())
#
# sns.heatmap(data, center=0, cmap="RdBu_r")
#
# plt.show()


size = 30000
x = scipy.arange(size)
y = scipy.int_(scipy.round_(scipy.stats.vonmises.rvs(5, size=size) * 47))
print(y)
h = plt.hist(y, bins=range(48), alpha=0.1, edgecolor='k')

dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']

for dist_name in dist_names:
    dist = getattr(scipy.stats, dist_name)
    param = dist.fit(y)
    pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size
    plt.plot(pdf_fitted, label=dist_name)
    plt.xlim(0, 47)
plt.legend(loc='upper right')
plt.show()
