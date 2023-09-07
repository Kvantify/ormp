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