import pytest

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.preprocessing import normalize

from ormp import OrderRecursiveMatchingPursuit

all_implementations = [
    "numpy",
    "jax-fast-runtime",
    "jax-fast-runtime-jit",
    "jax-fast-compilation-jit",
]


@pytest.mark.parametrize("implementation", all_implementations)
def test_hot_pursuit_raises_when_too_many_coefs_are_required(implementation):
    X = np.array([[1, 0], [0, 1]])
    y = np.array([3, 1])

    with pytest.raises(ValueError):
        OrderRecursiveMatchingPursuit(
            n_nonzero_coefs=3,
            fit_intercept=False,
            implementation=implementation,
        ).fit(X, y)


@pytest.mark.parametrize("implementation", all_implementations)
def test_empty_example_one_coef(implementation):
    X = np.array([[]])
    y = np.array([])

    with pytest.raises(ValueError):
        OrderRecursiveMatchingPursuit(
            n_nonzero_coefs=1,
            fit_intercept=False,
            implementation=implementation,
        ).fit(X, y)


@pytest.mark.parametrize("implementation", all_implementations)
def test_1d_example(implementation):
    X = np.array([[1]])
    y = np.array([3])
    reg_hp = OrderRecursiveMatchingPursuit(
        n_nonzero_coefs=1,
        fit_intercept=False,
        implementation=implementation,
    ).fit(X, y)
    assert np.all(reg_hp.coef_ == y)


@pytest.mark.parametrize("implementation", all_implementations)
def test_2d_example(implementation):
    X = np.array([[1, 0], [0, 1]])
    y = np.array([3, 1])
    reg_hp = OrderRecursiveMatchingPursuit(
        n_nonzero_coefs=2,
        fit_intercept=False,
        implementation=implementation,
    ).fit(X, y)
    assert np.all(reg_hp.coef_ == y)


@pytest.mark.parametrize("implementation", all_implementations)
def test_agreement_with_omp_on_simple_example(implementation):
    X, y = make_regression(n_samples=100, n_features=90, noise=4, random_state=0)
    X = normalize(X, norm="l2", axis=0)

    n_nonzero_coefs = 10
    reg_omp = OrthogonalMatchingPursuit(
        n_nonzero_coefs=n_nonzero_coefs, fit_intercept=False
    ).fit(X, y)
    reg_hp = OrderRecursiveMatchingPursuit(
        n_nonzero_coefs=n_nonzero_coefs,
        fit_intercept=False,
        implementation=implementation,
    ).fit(X, y)
    assert reg_hp.score(X, y) == pytest.approx(reg_omp.score(X, y))


@pytest.mark.parametrize("implementation", all_implementations)
def test_ten_percent_of_features_used_by_default(implementation):
    X, y = make_regression(n_samples=100, n_features=90, noise=4, random_state=0)
    X = normalize(X, norm="l2", axis=0)
    reg = OrderRecursiveMatchingPursuit(
        fit_intercept=False,
        implementation=implementation,
    ).fit(X, y)
    assert (reg.coef_ != 0).sum() == 9


@pytest.mark.parametrize("tol,n_nonzero_coefs", [(10000, 10), (1000, 21), (500, 51)])
def test_tolerance(tol, n_nonzero_coefs):
    X, y = make_regression(n_samples=100, n_features=90, noise=4, random_state=0)
    X = normalize(X, norm="l2", axis=0)
    reg = OrderRecursiveMatchingPursuit(fit_intercept=False, tol=tol).fit(X, y)
    assert (reg.coef_ != 0).sum() == n_nonzero_coefs
    y_hat = reg.predict(X)
    assert np.linalg.norm(y_hat - y) ** 2 < tol


def test_unachievable_tolerance():
    X, y = make_regression(n_samples=100, n_features=90, noise=4, random_state=0)
    X = normalize(X, norm="l2", axis=0)
    with pytest.raises(RuntimeError):
        OrderRecursiveMatchingPursuit(fit_intercept=False, tol=320).fit(X, y)


@pytest.mark.parametrize(
    "greediness,expected_residual",
    [(1, 1111), (2, 1216), (3, 1231), (4, 1478), (5, 1286)],
)
def test_greediness_non_divisible(greediness, expected_residual):
    X, y = make_regression(n_samples=100, n_features=90, noise=4, random_state=0)
    X = normalize(X, norm="l2", axis=0)
    n_nonzero_coefs = 17
    reg = OrderRecursiveMatchingPursuit(
        fit_intercept=False, n_nonzero_coefs=n_nonzero_coefs, greediness=greediness
    ).fit(X, y)
    assert (reg.coef_ != 0).sum() == n_nonzero_coefs
    residual = np.linalg.norm(reg.predict(X) - y) ** 2
    assert int(residual) == expected_residual
