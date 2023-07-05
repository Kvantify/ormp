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

    n_iter_ : int or array-like
        Number of active features across every target.

    n_nonzero_coefs_ : int
        The number of non-zero coefficients in the solution. If
        `n_nonzero_coefs` is None and `tol` is None this value is either set
        to 10% of `n_features` or 1, whichever is greater.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

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
        self, *, n_nonzero_coefs=None, tol=None, fit_intercept=True, greediness=1
    ):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.greediness = greediness
        if tol is not None:
            raise NotImplementedError("tol is not yet implemented")
        if fit_intercept:
            raise NotImplementedError(
                "the only implemented value of fit_intercept is False"
            )

    @staticmethod
    def _add_column(X, A_inv, u, dtype):
        X_prime = np.column_stack((X, u))
        w = X.T @ u
        A_inv_w = A_inv @ w
        # We assume X is normalized so u.T @ u = 1
        one_by_denominator = 1 / (1 - w.T @ A_inv_w)
        A_prime_inv_bottom_right = one_by_denominator
        n = A_inv.shape[0]
        if n == 0:
            A_prime_inv = np.array([[A_prime_inv_bottom_right]])
        else:
            numerator = np.outer(A_inv_w, A_inv_w.T)
            A_prime_inv_top_left = A_inv + numerator * one_by_denominator
            off_diag = -A_inv_w * one_by_denominator
            # The new A⁻¹ is a block matrix whose top left n x n block
            # is the original A⁻¹ matrix; it turns out that creating this
            # block matrix from an empty matrix is faster than using e.g. np.block
            A_prime_inv = np.empty((n + 1, n + 1), dtype=dtype)
            A_prime_inv[:n, :n] = A_prime_inv_top_left
            A_prime_inv[-1, :n] = off_diag
            A_prime_inv[:n, -1] = off_diag.T
            A_prime_inv[-1, -1] = A_prime_inv_bottom_right
        return X_prime, A_prime_inv

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
        # On our test cases, explicitly sticking to np.float64 instead of np.float32
        # yields a ~26% overall reduction in time to fit.
        dtype = np.float32

        # Ensure X is normalized
        n = X.shape[0]
        if np.any(np.abs(np.sum(X * X, axis=0) - 1) > 1e-7):
            raise ValueError("X must have normalized columns")
        np.testing.assert_allclose(np.ones(n), np.sum(X * X, axis=0))

        # Keep track of which column indices have been included so far
        remaining_indices = np.ones(n, dtype=np.bool_)
        X = X.astype(dtype)
        y = y.astype(dtype)
        X_curr = np.empty((n, 0), dtype=dtype)
        A_inv_curr = np.empty((0, 0), dtype=dtype)
        k = self.n_nonzero_coefs
        d = self.greediness
        while remaining_indices.sum() > n - k:
            pre_calc_proj = X_curr.T @ y
            us = X[:, remaining_indices]
            ws = X_curr.T @ us  # TODO: This can be simplified as we know X_prev.T @ us
            A_inv_ws = A_inv_curr @ ws
            w_t_A_inv_ws = np.sum(ws * A_inv_ws, axis=0)
            one_by_denominators = 1 / (1 - w_t_A_inv_ws)

            top_left_static = A_inv_curr
            top_left_dynamic = (
                np.einsum("jk, ik, j->ik", A_inv_ws, A_inv_ws, pre_calc_proj)
                * one_by_denominators
            )
            off_diags = -A_inv_ws * one_by_denominators[np.newaxis, :]

            proj_bottom = us.T @ y
            weight_top_left_static = top_left_static @ pre_calc_proj
            weight_top_left_dynamic = top_left_dynamic
            weight_top_left = (
                weight_top_left_static[:, np.newaxis] + weight_top_left_dynamic
            )
            weight_top_right = off_diags * proj_bottom
            weight_top = weight_top_left + weight_top_right

            weight_bottom_left = np.einsum("ij,i->j", off_diags, pre_calc_proj)
            weight_bottom_right = one_by_denominators * proj_bottom
            weight_bottom = weight_bottom_left + weight_bottom_right

            # The below alternative is almost faster, and gets rid of top_left_dynamic
            # above, which would be nice, but we keep the above for performance:
            #
            # weight_top_small = weight_top_left_static[:,np.newaxis] +\
            #      off_diags * proj_bottom
            # p1 = np.einsum(
            #     'mi, jk, ik, j->mk', X_curr, A_inv_ws,
            #     A_inv_ws, pre_calc_proj, optimize='optimal'
            # ) * one_by_denominators
            # p2 = X_curr @ weight_top_small
            # y_hats = p1 + p2 + us * weight_bottom

            y_hats = X_curr @ weight_top + us * weight_bottom

            diffs = y_hats - y[:, np.newaxis]
            all_evals = np.sum(diffs * diffs, axis=0)
            to_choose = np.argpartition(all_evals, d)[:d]

            evals = {
                p: all_evals[i] for i, p in enumerate(np.where(remaining_indices)[0])
            }
            sorted_evals = sorted(evals.items(), key=lambda x: x[1])
            to_choose = [x for x, _ in sorted_evals[:d]]
            for new_col in to_choose:
                X_curr, A_inv_curr = self._add_column(
                    X_curr, A_inv_curr, X[:, new_col], dtype=dtype
                )
            remaining_indices[to_choose] = 0

        # Build solution
        reg = LinearRegression(fit_intercept=False).fit(X[:, ~remaining_indices], y)
        self.coef_ = np.zeros(X.shape[1])
        self.coef_[~remaining_indices] = reg.coef_
        self.intercept_ = 0.0
        return self
