import os

import numpy as np


def random_unit_vector(n):
    a = np.random.normal(size=n)
    return a / np.linalg.norm(a)


def random_instance(n, m):
    X = np.empty((n, m), dtype=np.float64)
    for i in range(m):
        X[:, i] = random_unit_vector(n)
    y = random_unit_vector(n)
    return X, y


def check_file(filename):
    if os.path.exists(filename):
        raise RuntimeError(
            f"result file {filename} already exists; remove it before benchmarks"
        )
