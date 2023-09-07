"""Implementation of scikit-learn interface to the solvers."""
import functools
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


class OrderRecursiveMatchingPursuit(MultiOutputMixin, RegressorMixin, LinearModel):
    """Order Recursive Matching Pursuit (ORMP) model.

    Parameters
    ----------
    n_nonzero_coefs : int, default=None
        Desired number of non-zero entries in the solution. If None (by
        default) this value is set to 10% of n_features.

    tol : float, default=None
        Maximum squared norm of the residual. If not None, overrides n_nonzero_coefs.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    implementation : str, default="numpy"
        Which backend to use under the hood; allowed values are "numpy",
        "jax-fast-runtime", "jax-fast-runtime-jit", "jax-fast-compilation-jit".
        When using a "jax-" method, it is assumed that JAX is installed, and that
        any preconfiguration, such as installation of CUDA to make use of GPUs,
        has already been handled. Refer to the JAX installation guide at
        https://github.com/google/jax/#installation for possible options.

    greediness : int, default=1
        The number of columns to add in one iteration of the algorithm. Lower
        values will give better results but fits take longer.

    skip_validation : bool, default = False
        Skip all input validation checks (in particular, validation of input
        types and proper normalization of sample matrices); when using the
        JAX implementation, validation ends up taking a significant portion of
        the full running time, so consider making use of this attribute when
        you can guarantee beforehand that inputs are valid.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Parameter vector (w in the formula).

    intercept_ : float
        Independent term in decision function.

    Notes
    -----
    TODO: Reference for ORMP documentation.

    Examples
    --------
    >>> from hotpursuit.sklearn import OrderRecursiveMatchingPursuit
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.preprocessing import normalize
    >>> X, y = make_regression(noise=4, random_state=0)
    >>> X = normalize(X, norm="l2", axis=0)
    >>> reg = OrderRecursiveMatchingPursuit(n_nonzero_coefs=10, fit_intercept=False).fit(X, y)
    >>> reg.score(X, y)
    0.9991885378269406
    >>> reg.predict(X[:1,])
    array([-78.68765328])
    """

    _parameter_constraints: dict = {
        "n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0, None, closed="left"), None],
        "fit_intercept": ["boolean"],
        "implementation": [
            StrOptions(
                {
                    "numpy",
                    "jax-fast-runtime",
                    "jax-fast-runtime-jit",
                    "jax-fast-compilation-jit",
                }
            )
        ],
        "greediness": [Interval(Integral, 1, None, closed="left")],
        "skip_validation": ["boolean"],
    }

    def __init__(
        self,
        *,
        n_nonzero_coefs=None,
        tol=None,
        fit_intercept=True,
        implementation="numpy",
        greediness=1,
        skip_validation=False
    ):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.greediness = greediness
        self.implementation = implementation
        if tol is not None and implementation.startswith("jax"):
            raise NotImplementedError("tol is not yet implemented for the JAX solver")
        self.tol = tol
        if fit_intercept:
            raise NotImplementedError(
                "the only implemented value of fit_intercept is False"
            )
        self.fit_intercept = fit_intercept
        self.skip_validation = skip_validation

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
        if not self.skip_validation:
            X, y = self._validate_data(X, y, multi_output=True, y_numeric=True)
            # Ensure X is normalized
            if np.any(np.abs(np.sum(X * X, axis=0) - 1) > 1e-7):
                raise ValueError("X must have normalized columns")
        n_features = X.shape[1]
        if self.n_nonzero_coefs is None and self.tol is None:
            # default for n_nonzero_coefs is 0.1 * n_features
            # but at least one.
            self.n_nonzero_coefs_ = max(int(0.1 * n_features), 1)
        else:
            self.n_nonzero_coefs_ = self.n_nonzero_coefs

        if self.tol is None and self.n_nonzero_coefs_ > X.shape[1]:
            raise ValueError(
                "The number of atoms cannot be more than the number of features"
            )
        if self.implementation == "numpy":
            implementation = impl_np.ormp_numpy
        elif self.implementation.startswith("jax"):
            try:
                from . import impl_jax

                implementation = functools.partial(
                    impl_jax.hot_pursuit, implementation=self.implementation
                )
            except ImportError as e:
                raise RuntimeError(
                    "JAX not available; install JAX following the instructions on "
                    + "https://github.com/google/jax/#installation and for more "
                    + "information on why JAX is not included explicitly as a "
                    + "dependency, see (TODO: URL)."
                ) from e
        # TODO: Get actual weights instead of just indices from implementations.
        indices = implementation(
            X, y, self.n_nonzero_coefs_, tol=self.tol, greediness=self.greediness
        )
        # Build solution
        reg = LinearRegression(fit_intercept=False).fit(X[:, indices], y)
        self.coef_ = np.zeros(X.shape[1])
        self.coef_[indices] = reg.coef_
        self.intercept_ = 0.0
        return self
