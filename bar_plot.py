from cProfile import label
from turtle import color
from matplotlib import pyplot as plt
from matplotlib.axis import XAxis
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 16,
})
plt.grid(axis='y')

data_unfixed = [
    0,
    0,
    0,
    4,
    8,
    13,
    10,
    4,
    5,
    3,
    0,
]

data_fixed = [
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    4,
    7,
    7,
]

unfixed_sum = sum(data_unfixed)
fixed_sum = sum(data_fixed)

x_axis = np.arange(0, 11)

plt.bar(x_axis - 0.2, [d / unfixed_sum for d in data_unfixed], width=0.4, label='Original training', color="skyblue")
plt.bar(x_axis + 0.2, [d / fixed_sum for d in data_fixed], width=0.4, label='Improved training', color="pink")
plt.xticks(x_axis)
plt.ylabel('Fraction of runs')
plt.xlabel('Number of not-broken masks')

plt.legend()
plt.savefig('figures/bar_plot.pdf', bbox_inches='tight')

plt.show()