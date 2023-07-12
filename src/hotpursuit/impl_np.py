import numpy as np


def add_column(X, A_inv, u, dtype):
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


def eval_curr(X, A_inv, y):
    proj = X.T @ y
    coef = A_inv @ proj
    y_hat = X @ coef
    residual = y_hat - y
    return np.dot(residual, residual)


def hot_pursuit(X, y, n_nonzero_coefs, tol, greediness):
    # On our test cases, explicitly sticking to np.float64 instead of np.float32
    # yields a ~26% overall reduction in time to fit.
    dtype = np.float32
    # Keep track of which column indices have been included so far
    n, m = X.shape
    remaining_indices = np.ones(m, dtype=np.bool_)
    X = X.astype(dtype)
    y = y.astype(dtype)
    X_curr = np.empty((n, 0), dtype=dtype)
    A_inv_curr = np.empty((0, 0), dtype=dtype)
    k = n_nonzero_coefs
    # If tol is none, select columns until we have selected k of them;
    # otherwise select columns potentially forever (and break if we reach the
    # desired tolerance)
    while tol is not None or remaining_indices.sum() > m - k:
        d = greediness
        if tol is None:
            # Ensure that we do not take more columns than allowed
            # in case k is specified
            d = min(d, remaining_indices.sum() - (m - k))
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
        to_choose = np.argpartition(all_evals, d - 1)[:d]

        evals = {p: all_evals[i] for i, p in enumerate(np.where(remaining_indices)[0])}
        sorted_evals = sorted(evals.items(), key=lambda x: x[1])
        to_choose = [x for x, _ in sorted_evals[:d]]
        for new_col in to_choose:
            X_curr, A_inv_curr = add_column(
                X_curr, A_inv_curr, X[:, new_col], dtype=dtype
            )
        remaining_indices[to_choose] = 0

        if tol is not None:
            # Calculate the norm square of the residual now that all columns have
            # been added; in the special case where d == 1, we already have that from
            # our calculations above.
            current_objective = (
                eval_curr(X_curr, A_inv_curr, y) if d > 1 else np.min(all_evals)
            )
            if current_objective <= tol:
                break
            elif remaining_indices.sum() == 0:
                raise RuntimeError(
                    f"the given tolerance ({tol}) could not be achieved; "
                    f"the best achieved value was {current_objective}."
                )

    return ~remaining_indices
