import sys

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit

from ormp import OrderRecursiveMatchingPursuit

from .common import check_file, random_instance

np.random.seed(42)


def counts(n, m, n_nonzero_coefs, iters, greedinesses=[1]):
    draws = [0] * len(greedinesses)
    omp_wons = [0] * len(greedinesses)
    ormp_wons = [0] * len(greedinesses)

    for _ in range(iters):
        X, y = random_instance(m, n)
        coef1 = (
            OrthogonalMatchingPursuit(
                n_nonzero_coefs=n_nonzero_coefs, fit_intercept=False
            )
            .fit(X, y)
            .coef_
        )
        coefs = [
            OrderRecursiveMatchingPursuit(
                n_nonzero_coefs=n_nonzero_coefs, greediness=g, fit_intercept=False
            )
            .fit(X, y)
            .coef_
            for g in greedinesses
        ]

        for i, coef2 in enumerate(coefs):
            if np.allclose(coef1, coef2):
                draws[i] += 1
            elif np.linalg.norm(X @ coef1 - y) < np.linalg.norm(X @ coef2 - y):
                omp_wons[i] += 1
            elif np.linalg.norm(X @ coef1 - y) > np.linalg.norm(X @ coef2 - y):
                ormp_wons[i] += 1
            else:
                assert RuntimeError("could not determine winner")

    return (
        (draws, omp_wons, ormp_wons)
        if len(greedinesses) > 1
        else (draws[0], omp_wons[0], ormp_wons[0])
    )


if __name__ == "__main__":
    case = sys.argv[1]

    if case == "--small":
        check_file("accuracy.csv")
        for n_samples in range(3, 6):
            for n_features in range(3, n_samples + 1):
                for n_nonzero_coefs in range(2, n_features):
                    draw, omp_won, ormp_won = counts(
                        n_features, n_samples, n_nonzero_coefs, 10000
                    )
                    with open("accuracy.csv", "a") as f:
                        f.write(
                            f"{n_samples},{n_features},{n_nonzero_coefs},"
                            + f"{omp_won},{ormp_won},{draw}\n"
                        )

    elif case in ("--large", "--several"):
        greedinesses = [1] if case == "--large" else [2, 3, 4, 5]
        for greediness in greedinesses:
            check_file(f"accuracy-large-{greediness}.csv")
        for n_samples in (
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            125,
            150,
            175,
            200,
            225,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
        ):
            n_features = n_samples
            n_nonzero_coefs = n_features // 10
            draws, omp_wons, ormp_wons = counts(
                n_features, n_samples, n_nonzero_coefs, 10000, greedinesses
            )
            for greediness, draw, omp_won, ormp_won in zip(
                greedinesses, draws, omp_wons, ormp_wons
            ):
                print(greediness, n_samples, draw, omp_won, ormp_won)
                with open(f"accuracy-large-{greediness}.csv", "a") as f:
                    f.write(
                        f"{n_samples},{n_features},{n_nonzero_coefs},"
                        + f"{omp_won},{ormp_won},{draw}\n"
                    )

    elif case == "--various-nonzero":
        check_file("accuracy-various-nonzero.csv")
        n_samples = n_features = 100
        for n_nonzero_coefs in (
            list(range(1, 25, 2)) + [25, 27, 30, 33, 36] + list(range(40, 100, 5))
        ):
            draw, omp_won, ormp_won = counts(
                n_features, n_samples, n_nonzero_coefs, 10000
            )
            with open("accuracy-various-nonzero.csv", "a") as f:
                print(
                    f"{n_samples},{n_features},{n_nonzero_coefs},"
                    + f"{omp_won},{ormp_won},{draw}"
                )
                f.write(
                    f"{n_samples},{n_features},{n_nonzero_coefs},"
                    + f"{omp_won},{ormp_won},{draw}\n"
                )
