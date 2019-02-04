import numpy as np
import json
import sys


print(sys.path)
# dmp = json.loads(open("data/est.txt", 'r').read())
dmp = json.loads(open("data/tan_est.txt", 'r').read())

all_y = []

for x, y in dmp.items():
    all_y += list(map(lambda x: abs(x), y))
    print("x =", x)
    print("Mean", np.mean(y))
    # print("First quartile", np.percentile(y, 25))
    # print("second quartile", np.percentile(y, 50))
    # print("third quartile", np.percentile(y, 75))
    print(np.min(y))
    print(np.max(y))
    print()

print(all_y)
print("third quartile", np.percentile(all_y, 75))
