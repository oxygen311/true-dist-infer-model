import json
import numpy as np
from collections import defaultdict

xs = np.arange(0, 3, 0.01)


def get_cms_dist(file, max_cm=10):
    data = json.loads(open(file, 'r').read())
    # print("<File read (cms dist)>")
    cms_dist = []
    for x, ds in zip(xs, data):
        # print(x)
        cms = defaultdict(lambda: [])
        for n, _, cs in ds:
            for cm in range(2, max_cm):
                cms[cm].append(cs.get(str(cm), 0) / n)
        # print(cms)
        cms_dist.append(cms)
    return cms_dist
