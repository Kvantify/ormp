# hot-pursuit

This repository includes various Python implementations of the HOT pursuit algorithm for solving the sparse linear regression problem.

## Introduction

(TODO: Write about the problem this is solving)

## Installation

(TODO: Not yet true) The package is available on PyPI and can be installed using

```
pip install hot-pursuit
```

## Usage

In the example below, we generate a random example with 100 regressors and ask for a fit that uses at most 10 of them.

```python
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