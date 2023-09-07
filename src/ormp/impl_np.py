import numpy as np


def ormp_numpy(X, y, n_nonzero_coefs, tol=None, greediness=1):
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
