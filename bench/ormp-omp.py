import time

import numpy as np
from sklearn.linear_model._omp import orthogonal_mp

from ormp.impl_np import ormp_numpy

from .common import check_file, random_instance


np.random.seed(42)

if __name__ == "__main__":
    check_file("ormp-omp.csv")
    for n in (1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000):
        m = n
        k = n // 10
        d1 = []
        d2 = []
        for _ in range(100):
            X, y = random_instance(n, m)
            t0 = time.time()
            ormp_numpy(X, y, k)
            d1.append(time.time() - t0)
            t0 = time.time()
            orthogonal_mp(X, y, n_nonzero_coefs=k)
            d2.append(time.time() - t0)
        with open("ormp-omp.csv", "a") as f:
            f.write(f"{n},{np.mean(d1)},{np.std(d1)},{np.mean(d2)},{np.std(d2)}\n")
