# Python Lasso/ElasticNet/Ridge Regression with Customized Penalties
An extension of sklearn's Lasso/ElasticNet/Ridge model to allow users to customize the penalties of different covariates. Works similar to `penalty.factor` parameter in R's glmnet.

## Introduction
Sometimes we have prior knowledge that some covariates are important and some are not. Like weekend and holiday should be strong predictors to daily traffic flow, gender should be a strong predictor to breast cancer risk, whereas ice-cream sales should not contribute to crime rate. In such cases, when doing regularized linear regression, we want to penalize certain covariates with different weights.

This module also allows one to do basic two-step adaptive regularized regression.

## Usage
Two classes, CustomENet and CustomENetCV, are provided for regression and cross-validation along the regularization path. Both accept an additional penalty weight parameter for fit. See the example notebook and docstrings for details.

## Algorithm
Let me explain with Lasso for simplicity. The extension to ElasticNet and Ridge is trivial.
In regular Lasso we penalize each covariate with equal weight, i.e., 1. Now we define a vector $s$ to store the customized penalty weights of covariates. So the objective is to minimize
$$\frac{1}{2n}||y-X\beta||_2^2 + \alpha||s \odot \beta||_1$$
where $\odot$ refers to element-wise product and $n$ is the number of samples.
We can further classify $s_i$ into three categories:

1. $s_i>0$ and is finite.
2. $s_i=0$, do not at all penalize $\beta_i$
3. $s_i=+\infty$, set  $\beta_i = 0$

For 3 we simply remove the corresponding covariates. For 1 & 2 we split $X$ and $\beta$ into submatrices based on $s_i$'s type, and the objective function turns to:
$$\frac{1}{2n}||y-X_1\beta_1-X_2\beta_2||_2^2 + \alpha||s_1 \odot \beta_1||_1$$
Let $\beta_1' = s_1 \odot \beta_1$, then $\beta_1 = \beta_1' \oslash s_1$ where $\oslash$ is element-wise division. Substitute it to the objective function gives
$$\frac{1}{2n}||y-X_1 (\beta_1' \oslash s_1) - X_2\beta_2||_2^2 + \alpha||\beta_1'||_1$$
which equals to 
$$\frac{1}{2n}||y- X_1' \beta_1' - X_2\beta_2||_2^2 + \alpha||\beta_1'||_1;\ X_1' = X_1 \oslash (\vec{1} s_1^T)$$
where $\vec{1}$ is vector of ones with proper shape.
Now do orthogonal decompositon to the expression inside $||\cdot||_2^2$ based on the column space of $X_2$. Let $X_2^+$ be the pseudo inverse of $X_2$, then $H_2 = X_2 X_2^+$ is the orthogonal projector to $Col(X_2)$. This gives:
 $$\frac{1}{2n}(||H_2 (y- X_1' \beta_1') - X_2\beta_2||_2^2 + ||(1-H_2)(y- X_1' \beta_1')||_2^2) + \alpha||\beta_1'||_1$$
Note that $H_2 (y- X_1' \beta_1') - X_2\beta_2 = 0$ when $\beta_2 = X_2^+(y- X_1' \beta_1')$ for arbitrary $\beta_1'$. Actually $\beta_2$ is an OLS solution of $X_2\beta_2 = (y- X_1' \beta_1')$. So the objective is equivalent to minimize:
 $$\frac{1}{2n}||(1-H_2)(y- X_1' \beta_1')||_2^2 + \alpha||\beta_1'||_1$$
which can be solved with regular Lasso. After getting $\beta_1'$, as mentioned above, we can calculate $\beta_2 = X_2^+(y- X_1' \beta_1')$ and $\beta_1 = \beta_1' \oslash s_1$.

## Acknowledgement
The algorithm is inspired by this answer:
https://stats.stackexchange.com/a/307133/68424
