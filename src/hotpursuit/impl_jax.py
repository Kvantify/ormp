"""Implementation of HOT Pursuit using JAX."""
import jax
import jax.numpy as jnp
import numpy as np


def ormp_fast_runtime(X, y, k):
    X = X.astype(jnp.float32)
    y = y.astype(jnp.float32)
    n, m = X.shape
    xt = X.T
    xtbp = xt @ y
    alpnormsq = jnp.ones(m, dtype=jnp.float32)
    selected = jnp.zeros(m, dtype=jnp.bool_)
    proj = jnp.zeros((n, n), dtype=jnp.float32)
    for p in range(k):
        allobjs = jnp.divide(xtbp**2, alpnormsq)
        allobjs = jnp.where(selected, 0, allobjs)
        best_l = jnp.argmax(allobjs)
        selected = selected.at[best_l].set(True)
        al = X[:, best_l]
        proj_ort_al = al - proj @ al
        proj_ort_al /= jnp.sqrt(jnp.dot(proj_ort_al, proj_ort_al))
        xt_proj_ort_al = xt @ proj_ort_al
        alpnormsq -= xt_proj_ort_al**2
        outer = jnp.outer(proj_ort_al, proj_ort_al)
        proj += outer
        xtbp -= jnp.dot(proj_ort_al, y) * xt_proj_ort_al
        p += 1
    return selected


ormp_fast_runtime_jit = jax.jit(ormp_fast_runtime, static_argnames=("k",))


def ormp_fast_compilation(X, y, k):
    X = X.astype(jnp.float32)
    y = y.astype(jnp.float32)
    n, m = X.shape
    xt = X.T
    xtbp = xt @ y
    alpnormsq = jnp.ones(m, dtype=jnp.float32)
    selected = jnp.zeros(m, dtype=jnp.bool_)
    projs = jnp.zeros((k, n), dtype=jnp.float32)

    def body_fun(p, val):
        xtbp, alpnormsq, selected, projs = val

        allobjs = jnp.divide(xtbp**2, alpnormsq)
        allobjs = jnp.where(selected, 0, allobjs)
        best_l = jnp.argmax(allobjs)
        selected = selected.at[best_l].set(True)
        al = X[:, best_l]

        proj_ort_al = al - projs.T @ (projs @ al)
        proj_ort_al /= jnp.sqrt(jnp.dot(proj_ort_al, proj_ort_al))
        xt_proj_ort_al = xt @ proj_ort_al
        alpnormsq -= xt_proj_ort_al**2
        projs = projs.at[p].set(proj_ort_al)
        xtbp -= jnp.dot(proj_ort_al, y) * xt_proj_ort_al

        return xtbp, alpnormsq, selected, projs

    _, _, selected, _ = jax.lax.fori_loop(
        0, k, body_fun, (xtbp, alpnormsq, selected, projs)
    )
    return selected


ormp_fast_compilation_jit = jax.jit(ormp_fast_compilation, static_argnames=("k",))


implementations = {
    "jax-fast-runtime": ormp_fast_runtime,
    "jax-fast-runtime-jit": ormp_fast_runtime_jit,
    "jax-fast-compilation-jit": ormp_fast_compilation_jit,
}


def hot_pursuit(X, y, n_nonzero_coefs, implementation, tol, greediness):
    if greediness != 1:
        return NotImplementedError(
            "for the JAX implementation, only greediness=1 is supported"
        )
    if tol is not None:
        return NotImplementedError(
            "for the JAX implementation, only tol=None is supported"
        )
    return implementations[implementation](X, y, n_nonzero_coefs)
