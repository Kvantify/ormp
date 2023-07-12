"""Implementation of HOT Pursuit using JAX."""
import jax
import jax.numpy as jnp
import numpy as np


def add_column(X, A_inv, u):
    X_prime = jnp.column_stack((X, u))
    w = X.T @ u
    A_inv_w = A_inv @ w
    # We assume X is normalized so u.T @ u = 1
    one_by_denominator = 1 / (1 - w.T @ A_inv_w)
    A_prime_inv_bottom_right = jnp.array([[one_by_denominator]])
    n = A_inv.shape[0]
    if n == 0:
        A_prime_inv = A_prime_inv_bottom_right
    else:
        numerator = jnp.outer(A_inv_w, A_inv_w.T)
        A_prime_inv_top_left = A_inv + numerator * one_by_denominator
        off_diag = jnp.array([-A_inv_w * one_by_denominator])
        A_prime_inv = jax.numpy.block(
            [[A_prime_inv_top_left, off_diag.T], [off_diag, A_prime_inv_bottom_right]]
        )
    return X_prime, A_prime_inv


def find_next_state(X_curr, A_inv_curr, y, us):
    pre_calc_proj = X_curr.T @ y
    ws = X_curr.T @ us  # Can be simplified as we know X_prev.T @ us
    A_inv_ws = A_inv_curr @ ws
    w_t_A_inv_ws = jnp.sum(ws * A_inv_ws, axis=0)
    one_by_denominators = 1 / (1 - w_t_A_inv_ws)

    top_left_static = A_inv_curr
    top_left_dynamic = (
        jnp.einsum("jk, ik, j->ik", A_inv_ws, A_inv_ws, pre_calc_proj)
        * one_by_denominators
    )
    off_diags = -A_inv_ws * one_by_denominators[jnp.newaxis, :]

    proj_bottom = us.T @ y
    weight_top_left_static = top_left_static @ pre_calc_proj
    weight_top_left_dynamic = top_left_dynamic
    weight_top_left = weight_top_left_static[:, jnp.newaxis] + weight_top_left_dynamic
    weight_top_right = off_diags * proj_bottom
    weight_top = weight_top_left + weight_top_right

    weight_bottom_left = jnp.einsum("ij,i->j", off_diags, pre_calc_proj)
    weight_bottom_right = one_by_denominators * proj_bottom
    weight_bottom = weight_bottom_left + weight_bottom_right

    y_hats = X_curr @ weight_top + us * weight_bottom

    diffs = y_hats - y[:, jnp.newaxis]
    all_evals = jnp.sum(diffs * diffs, axis=0)

    # Take best
    new_col = jnp.argmin(all_evals)
    X_curr, A_inv_curr = add_column(X_curr, A_inv_curr, us[:, new_col])
    return X_curr, A_inv_curr, new_col


def hot_pursuit_impl(X, y, n_nonzero_coefs):
    n = X.shape[0]
    X_curr = jnp.empty((n, 0))
    A_inv_curr = jnp.empty((0, 0))
    us = X
    k = n_nonzero_coefs
    chosen_indices = jnp.empty(k, dtype=jnp.int32)
    # TODO: Could this be a jax.lax.fori_loop?
    for i in range(k):
        X_curr, A_inv_curr, new_col = find_next_state(X_curr, A_inv_curr, y, us)
        # Remove the new_col'th column from us
        us = jnp.where(jnp.arange(us.shape[1] - 1) < new_col, us[:, :-1], us[:, 1:])
        # Note that in particular, the below alternative wouldn't work a priori:
        #    us = jnp.delete(us, new_col, axis=1)
        chosen_indices = chosen_indices.at[i].set(new_col)
    return chosen_indices


hot_pursuit_impl_jit = jax.jit(hot_pursuit_impl, static_argnames=("n_nonzero_coefs",))


def hot_pursuit(X, y, n_nonzero_coefs, tol, greediness, use_jit):
    if greediness != 1:
        return NotImplementedError(
            "for the JAX implementation, only greediness=1 is supported"
        )
    if tol is not None:
        return NotImplementedError(
            "for the JAX implementation, only tol=None is supported"
        )
    impl = hot_pursuit_impl_jit if use_jit else hot_pursuit_impl
    removed_columns = impl(X, y, n_nonzero_coefs)
    indices = np.zeros(X.shape[1], dtype=np.bool_)
    to_take = list(range(X.shape[1]))
    for index in removed_columns:
        indices[to_take.pop(index)] = True
    return indices
