"""Implementation of scikit-learn interface to the solvers."""
from numbers import Integral, Real

import numpy as np
from sklearn.base import MultiOutputMixin, RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.linear_model import LinearRegression
from sklearn.utils._param_validation import (
    Interval,
    StrOptions,
)

from . import impl_np


class HotPursuit(MultiOutputMixin, RegressorMixin, LinearModel):
    """HOT Pursuit model.

    Parameters
    ----------
    n_nonzero_coefs : int, default=None
        Desired number of non-zero entries in the solution. If None (by
        default) this value is set to 10% of n_features.

    tol : float, default=None
        Maximum norm of the residual. If not None, overrides n_nonzero_coefs.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    greediness : int, default=1
        The number of columns to add in one iteration of the algorithm. Lower
        values will give better results but fits take longer.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the formula).

    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function.

    Notes
    -----
    TODO: Reference for HOT pursuit documentation.

    Examples
    --------
    >>> from hotpursuit.sklearn import HotPursuit
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.preprocessing import normalize
    >>> X, y = make_regression(noise=4, random_state=0)
    >>> X = normalize(X, norm="l2", axis=0)
    >>> reg = HotPursuit(n_nonzero_coefs=10, fit_intercept=False).fit(X, y)
    >>> reg.score(X, y)
    0.9991885378269406
    >>> reg.predict(X[:1,])
    array([-78.68765328])
    """

    _parameter_constraints: dict = {
        "n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0, None, closed="left"), None],
        "fit_intercept": ["boolean"],
        "implementation": [StrOptions({"numpy", "jax"})],
        "greediness": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        *,
        n_nonzero_coefs=None,
        tol=None,
        fit_intercept=True,
        implementation="numpy",
        greediness=1
    ):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.greediness = greediness
        self.implementation = implementation
        if tol is not None:
            raise NotImplementedError("tol is not yet implemented")
        if fit_intercept:
            raise NotImplementedError(
                "the only implemented value of fit_intercept is False"
            )

    def fit(self, X, y):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X, y = self._validate_data(X, y, multi_output=True, y_numeric=True)

        # Ensure X is normalized
        if np.any(np.abs(np.sum(X * X, axis=0) - 1) > 1e-7):
            raise ValueError("X must have normalized columns")

        if self.implementation == "numpy":
            implementation = impl_np.hot_pursuit
            indices = impl_np.hot_pursuit
        elif self.implementation == "jax":
            try:
                from . import impl_jax

                implementation = impl_jax.hot_pursuit
            except ImportError:
                raise RuntimeError(
                    "JAX not available; install JAX following the instructions on "
                    + "https://github.com/google/jax/#installation and for more "
                    + "information on why JAX is not included explicitly as a "
                    + "dependency, see (TODO: URL)."
                )
        # TODO: Get actual weights instead of just indices from implementations.
        indices = implementation(X, y, self.n_nonzero_coefs, self.greediness)
        # Build solution
        reg = LinearRegression(fit_intercept=False).fit(X[:, indices], y)
        self.coef_ = np.zeros(X.shape[1])
        self.coef_[indices] = reg.coef_
        self.intercept_ = 0.0
        return self
