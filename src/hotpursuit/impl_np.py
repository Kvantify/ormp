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
    dtype = np.float32
    X = X.astype(dtype)
    y = y.astype(dtype)
    # Keep track of which column indices have been included so far
    n, m = X.shape
    xt = X.T
    xtbp = xt @ y
    k = n_nonzero_coefs
    alpnormsq = np.ones(m, dtype=dtype)
    unselected = np.ones(m, dtype=np.bool_)
    projs = np.empty((k if tol is None else n, n), dtype=dtype)
    p = 0
    # If tol is none, select columns until we have selected k of them;
    # otherwise select columns potentially forever (and break if we reach the
    # desired tolerance)
    while True:
        if tol is not None and p == m:
            raise RuntimeError(f"the given tolerance ({tol}) could not be achieved")

        d = greediness
        if tol is None:
            # Ensure that we do not take more columns than allowed
            # in case k is specified
            d = min(d, k - p)

        allobjs = np.divide(
            xtbp**2, alpnormsq, out=np.zeros(m, dtype=dtype), where=unselected
        )
        to_choose = np.argpartition(-allobjs, d - 1)[:d]
        for l in to_choose:
            unselected[l] = False
            if tol is None and p == k - 1:
                return ~unselected
            al = X[:, l]
            proj_ort_al = al - projs[:p].T @ (projs[:p] @ al)
            proj_ort_al /= np.sqrt(np.dot(proj_ort_al, proj_ort_al))
            projs[p] = proj_ort_al
            p += 1
            if tol is not None:
                # To check if we are within the desired tolerance, we do
                # explicitly calculate the objective. This can likely be
                # done much more efficiently.
                y_hat = projs[:p].T @ (projs[:p] @ y) - y
                score = np.dot(y_hat, y_hat)
                if score < tol:
                    return ~unselected
            xt_proj_ort_al = xt @ proj_ort_al
            alpnormsq -= xt_proj_ort_al**2
            xtbp -= np.dot(proj_ort_al, y) * xt_proj_ort_al
