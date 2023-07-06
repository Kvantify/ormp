import pytest

from sklearn.datasets import make_regression
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.preprocessing import normalize

from hotpursuit.hotpursuit import HotPursuit


@pytest.mark.parametrize("implementation", ["numpy", "jax"])
def test_agreement_with_omp_on_simple_example(implementation):
    X, y = make_regression(noise=4, random_state=0)
    X = normalize(X, norm="l2", axis=0)

    n_nonzero_coefs = 10
    reg_omp = OrthogonalMatchingPursuit(
        n_nonzero_coefs=n_nonzero_coefs, fit_intercept=False
    ).fit(X, y)
    reg_hp = HotPursuit(
        n_nonzero_coefs=n_nonzero_coefs,
        fit_intercept=False,
        implementation=implementation,
    ).fit(X, y)
    print(reg_omp.coef_)
    print(reg_hp.coef_)
    assert reg_hp.score(X, y) == pytest.approx(reg_omp.score(X, y))
