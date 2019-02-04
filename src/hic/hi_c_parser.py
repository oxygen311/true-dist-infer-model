import pandas as pd
from bisect import bisect
import numpy as np

file_name = "data/hi-c.txt"


def my_round(x, base):
    return int(base * round(float(x)/base))


class HiCData:
    def __init__(self, base):
        self.data = pd.read_csv(file_name, sep="\t", header=None)
        self.data.columns = ["x", "y", "o/e"]
        self.ys = sorted(list(set(self.data['y'].values)))
        self.base = base

    def get_nearest_value(self, x, y):
        if x > y:
            x, y = y, x
        rx = my_round(x, self.base)
        ry = my_round(y, self.base)
        # ry = HiCData.get_nearest_in_array(self.ys, y)
        # xs = sorted(self.data.loc[self.data['y'] == ry]['x'].values)
        # rx = HiCData.get_nearest_in_array(xs, x)
        return self.data.loc[(self.data['x'] == rx) & (self.data['y'] == ry)]['o/e'].values[0]

    def get_mean_v(self):
        return np.mean([row['o/e'] for _, row in self.data.iterrows()])

    # @staticmethod
    # def get_nearest_in_array(bs, b):
    #     i = bisect(bs, b)
    #     if i == 0:
    #         print("WARNING", i, len(bs), b, bs)
    #         return bs[i]
    #     if i == len(bs):
    #         print("WARNING", abs(bs[-1] - b), i, len(bs), b, bs)
    #         return bs[-1]
    #     else:
    #         return bs[i] if abs(bs[i] - b) < abs(bs[i - 1] - b) else bs[i - 1]
