# ORMP

This repository includes various Python implementations of the Order Recursive Matching Pursuit (ORMP) algorithm for solving the sparse linear regression problem.

## Introduction

Here, we aim to solve various versions of sparse linear regression, also known in the signal processing community as sparse approximation, sparse representation, compressed sensing or signal reconstruction; other algorithms include [matching pursuit](https://en.wikipedia.org/wiki/Matching_pursuit) and its variants, such as orthogonal matching pursuit (OMP), regularized OMP, stagewise OMP, compressed sensinting matching pursuit, and subspace pursuit.

The problem we aim to solve is the following: Given an $n$-dimensional vector $y \in \mathbb{R}^n$ of observed values, and an $n \times p$-dimensional matrix $X$ of regressors, find parameters $\beta \in \mathbb{R}^p$ so that $y = X\beta + \varepsilon$, with $\varepsilon$ as small as possible; this is nothing but linear regression, but we also want to require that $\beta$ has only few non-zero entries.

Specifically, let $S_k = \lbrace \beta \in \mathbb{R}^p \mid \lVert \beta \rVert_0 \leq k \rbrace$ be the set of vectors in $\mathbb{R}^p$ with at most $k$ non-zero entries. We then consider the problem of determining

$$\min_{\beta \in S_k} \lVert X\beta - y \rVert_2,$$

for a fixed allowed number of non-zero entries $k \geq 0$. We also consider the corresponding problem of determining the smallest value of $k$ to achieve a given tolerance $T \geq 0$ on the residual vector $\varepsilon$,

$$\min \lbrace k \in \mathbb{N} \mid \exists \beta \in S_k : \lVert X\beta - y \rVert_2 \leq T \rbrace.$$

We do not solve the problems to optimality but instead provide a greedy heuristic which iteratively adds non-zero entries (hence features to the linear model). This is similar to what happens in matching pursuit, where one iteratively picks the column with maximum inner product with the current residual; here, instead, we choose the column which minimizes the actual objective function (the length of the projection of $y$ on the subspace obtained by adding the column); generally speaking, this can be seen as a more thorough greedy approach than matching pursuit, which will, at least in cases of interest to us, yield better results but generally at the cost of spending more time to get those results.

The solvers come with implementations of the scikit-learn estimator API and therefore can be used as a part of scikit-learn pipelines. We provide an implementation that relies only on NumPy, as well as an implementation that makes use of JAX to be able to make efficient use of any available GPUs/TPUs.

## Installation

(TODO: Not yet true) The package is available on PyPI and can be installed using

```
pip install ormp
```

As the best possible configuration of JAX depends on the system the code will be running on, and may include things like CUDA being readily available when running the solver on a GPU, JAX is not included as a dependency and to make use of the JAX-based solvers, it must be installed separately. Refer to the [JAX installation guide](https://github.com/google/jax/#installation) for possible options. If you only wish to quickly get a feel for the performance of ORMP, you can start by installing it without JAX and using only the already efficient NumPy-based solver.

## Usage

In the example below, we generate a random example with 100 regressors and ask for a fit that uses at most 10 of them.

```python
>>> from ormp import OrderRecursiveMatchingPursuit
>>> from sklearn.datasets import make_regression
>>> from sklearn.preprocessing import normalize
>>> X, y = make_regression(noise=4, random_state=0)
>>> X = normalize(X, norm="l2", axis=0)
>>> reg = OrderRecursiveMatchingPursuit(n_nonzero_coefs=10, fit_intercept=False).fit(X, y)
>>> reg.score(X, y)
0.9991885378269406
>>> reg.predict(X[:1,])
array([-78.68765328])
>>> reg.coef_
array([  0.        ,   0.        ,  92.43353271,   0.        ,
         0.        ,   0.        ,   0.        ,  77.50286865,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        , 713.07751465,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        , 382.94140625,   0.        ,   0.        ,
       527.96832275,   0.        , 444.06378174,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
       286.8444519 ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        , 646.28283691,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
       648.30822754,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        ,   0.        ,   0.        ,   0.        ,
         0.        , 141.12867737,   0.        ,   0.        ])
```

### Choosing an implementation

As stated above, we provide both non-JAX and several JAX based solvers. Which one to choose depends on characteristics of the input data, whether one can live with a long first initial compilation time, and whether one will be running the solver on inputs with many different shapes, each of which will require recompilation.

To accommodate these different scenarios, we include separate JAX solvers that are optimized for runtime and compilation time.

For example, when run on a laptop with a GPU, fitting a `(1000, 1000)`-shaped sample while requiring no more than 100 non-zero coefficients, a typical picture is that

* `ormp.impl_np.ormp_numpy` will take 30 ms to run
* `ormp.impl_jax.ormp_fast_runtime_jit` will take 23 seconds to compile, then 3 ms to run,
* `ormp.impl_jax.ormp_fast_compilation_jit` will take 180 ms to compile, then 15 ms to run.

For comparison, `sklearn.linear_model.OrthogonalMatchingPursuit.fit` would take about 80 ms on such an input on such a device.


## Comparison with Orthogonal Matching Pursuit

It is illustrative to compare the behavior of the algorithm with that of Orthogonal Matching Pursuit (OMP), [as available in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html#sklearn.linear_model.OrthogonalMatchingPursuit).

As mentioned above, ORMP can be considered as a variation on the iterative approach defining OMP, in that both methods are iterative, but here we spend more time in each iteration, picking the input vector that minimizes the residual rather than simply the input vector with the smallest angle with the residual.

Both algorithms being heuristics, it is not clear a priori how much of a benefit this will give on a given input, if any; it is easy to imagine that for certain classes of inputs, one algorithm is a better choice than the other, and certainly our own work here has been inspired by ORMP greatly outperforming OMP on inputs of interest.

One way to get a bit of an intuition about how the algorithms compare is by running them on random small instances. Here, specifically, we run them on 10,000 inputs consisting of columns sampled uniformly on the unit spheres of various dimensions, and vary the number of desired non-zero coefficients; we then compare the fits and keep track of which algorithm led to the better fit:

|   n_samples |   n_features |   n_nonzero_coefs |   OMP wins |   ORMP wins |   Draw |
|-------------|--------------|-------------------|------------|-------------|--------|
|           3 |            3 |                 2 |          0 |        1342 |   8658 |
|           4 |            3 |                 2 |          0 |         761 |   9239 |
|           4 |            4 |                 2 |          0 |        1248 |   8752 |
|           4 |            4 |                 3 |        262 |        1999 |   7739 |
|           5 |            3 |                 2 |          0 |         513 |   9487 |
|           5 |            4 |                 2 |          0 |         923 |   9077 |
|           5 |            4 |                 3 |        145 |        1355 |   8500 |
|           5 |            5 |                 2 |          0 |        1189 |   8811 |
|           5 |            5 |                 3 |        276 |        2189 |   7535 |
|           5 |            5 |                 4 |        572 |        2674 |   6754 |

Note, for instance, that when `n_nonzero_coefs` is 2, OMP never wins. This is the case since the two methods will always pick the same first vector, and ORMP will then always pick the best of the remaining ones. It is however, not the case that ORMP is always the strictly better choice, and for larger instances, OMP starts winning a significant amount of time.

The code to generate a row in this table is given below:

```python
def random_unit_vector(n):
    a = np.random.normal(size=n)
    return a / np.linalg.norm(a)

def random_instance(n, m):
    X = np.empty((n, m), dtype=np.float64)
    for i in range(m):
        X[:, i] = random_unit_vector(n)
    y = random_unit_vector(n)
    return X, y
        
def counts(n, m, n_nonzero_coefs, iters):
    draw = omp_won = ormp_won = 0
    for _ in range(iters):
        X, y = random_instance(m, n)
        coef1 = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, fit_intercept=False).fit(X, y).coef_
        coef2 = OrderRecursiveMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, fit_intercept=False).fit(X, y).coef_
        if np.allclose(coef1, coef2):
            draw += 1
        elif np.linalg.norm(X @ coef1 - y) < np.linalg.norm(X @ coef2 - y):
            omp_won += 1
        elif np.linalg.norm(X @ coef1 - y) > np.linalg.norm(X @ coef2 - y):
            ormp_won += 1
        else:
            assert False
    return draw, omp_won, ormp_won

counts(5, 5, 4, 10_000)
```