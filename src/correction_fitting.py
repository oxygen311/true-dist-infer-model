import numpy as np
import matplotlib.pyplot as plt
import math

step_correction = lambda t: - 0.0045 - 0.0015 + (- 0.002 if t <= 0.7 else 0) + (- 0.002 if t <= 0.5 else 0)
aprox_correction = lambda t: (1 / (1 + math.exp((-t + 0.6) * 20))) / 250 - 0.01


ts = np.arange(0, 1, 0.01)
cs = [step_correction(t) for t in ts]
cs2 = [aprox_correction(t) for t in ts]

plt.plot(ts, cs)
plt.plot(ts, cs2)
plt.savefig("correction_fitting.pdf")
plt.show()