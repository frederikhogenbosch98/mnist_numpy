import numpy as np


def normalize(x):
    return 1 / (1 + np.exp(-x))


def forward(a, b):
    return a @ b