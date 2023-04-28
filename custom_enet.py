# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.linear_model._coordinate_descent import _alpha_grid as alpha_grid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from statsmodels.api import add_constant
from typing import Union

class CustomENet(BaseEstimator):
    
    """ElasticNet with customized penalties of covariates. Most paramters are 
    the same as sklearn's ElasticNet. Parameters listed here are either new
    or need to be checked.
    
    Parameters
    ----------
    alpha:
        Coefficient of the penalty terms. Note that for Ridge, this alpha does 
        not have the same magnitude as the alpha of sklearn's Ridge due to the
        difference of objective functions. Actually, 
        alpha(sklearn's Ridge) = n_samples * alpha(here)
    l1_ratio: 
        0 for Ridge, (0, 1) for ElasticNet, 1 for Lasso. Avoid (0, 0.01) if 
        possible.
    fit_intercept: 
        Set this to False if either:
            1. X already has a non-zero constant column. In this case make sure to set the corresponding penalty to 0.
            2. standardize = True and y is centered.
            3. standardize = False but all columns of X and y are centered.
    standardize:
        Whether to standardize X before fit. Only covariates with finite positive 
        penalties will be standardized.
        
    Attributes
    ----------
    w:
        Regression coefficients. If fit_intercept=True, w[-1] is the intercept.
    model:
        Sklearn's model used in the algorithm. Note this is an intermediate model
        and should not be used for prediction.
    """
    
    def __init__(self, alpha = 1.0, 
                 l1_ratio = 0.5,
                 standardize = True,
                 fit_intercept = True,
                 positive = False,
                 max_iter=2000, tol=1e-4, 
                 warm_start = False, 
                 precompute = False,
                 selection = 'cyclic',
                 random_state = 0
                 ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.positive = positive
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.precompute = precompute
        self.random_state = random_state
        self.selection = selection
        self.fitted = False
        
    def preprocess_(self, X: np.ndarray, 
                    y: np.ndarray, 
                    s = Union[None, np.ndarray]):
        X = X.copy()
        y = y.copy().reshape(-1, 1)
        if s is None:
            s = np.ones(X.shape[1])
        else:
            s = s.copy().ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError('y should be a 1-D vector with length equal to X.shape[0]')
        if X.shape[1] != s.shape[0]:
            raise ValueError('s should be a 1-D vector with length equal to X.shape[1]')
        if not np.all(s >= 0):
            raise ValueError('all elements in s must be non-negative')
        if self.fit_intercept:
            s = np.append(s, .0)
            X = add_constant(X, prepend=False)
        wf = ~(np.isinf(s) | (s == .0))
        if self.standardize:
            x_scaler = StandardScaler()
            X[:, wf] = x_scaler.fit_transform(X[:, wf])
            self.x_scaler = x_scaler
        else:
            self.x_scaler = None
        self.nvar = X.shape[1]
        self.X = X
        self.y = y
        self.s = s
        self.wf = wf
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray, 
            s = Union[None, np.ndarray]
            ):
        
        """Fit model with customized penalty. 
        
        
        Parameters
        ----------
        X : 
            Array with shape (n_samples, n_features).
        y : 
            Array with shape (n_samples,) or (n_samples, 1). Currently 
            multitarget is not suppoerted.
        s: 
            Penalty weight vector with shape (n_features,) or (n_features, 1).
            Weight must be non-negative but can be infinite.
        """
        
        self.preprocess_(X, y, s)
        
        nsample = X.shape[0]
        sf_1 = self.wf
        sf_2 = self.s == .0
        X_1 = self.X[:, sf_1]
        s_1 = self.s[sf_1]
        X_2 = self.X[:, sf_2]
        
        X_1p = X_1 / s_1.reshape(1, -1)
        X_2i = np.linalg.pinv(X_2)
        H_2 = X_2 @ X_2i
        Q_2 = (np.identity(nsample) - H_2)
        y_q = np.asfortranarray(Q_2 @ self.y)
        X_1q = np.asfortranarray(Q_2 @ X_1p)
        
        curr_model_type = Ridge if self.l1_ratio == 0 else ElasticNet
        if (not self.fitted) or (not isinstance(self.model, curr_model_type)):
            if self.l1_ratio != 0:
                self.model = ElasticNet(fit_intercept=False)
            else:
                self.model = Ridge(fit_intercept=False)
        if self.l1_ratio != 0:
            self.model.set_params(alpha=self.alpha, 
                                  l1_ratio = self.l1_ratio, 
                                  positive=self.positive,
                                  precompute=self.precompute, 
                                  warm_start=self.warm_start,
                                  max_iter=self.max_iter, tol=self.tol,
                                  selection = self.selection,
                                  random_state=self.random_state)
        else:
            self.model.set_params(alpha=nsample*self.alpha, max_iter=self.max_iter, 
                                  tol=self.tol, positive=self.positive, 
                                  random_state=self.random_state)
        self.model.fit(X_1q, y_q)
        beta_1p = self.model.coef_.ravel()
        beta_1 = beta_1p / s_1
        beta_2 = (X_2i @ (self.y - X_1p @ beta_1p.reshape(-1,1))).ravel()
        w = np.zeros(self.nvar)
        w[sf_1] = beta_1
        w[sf_2] = beta_2
        self.w = w
        self.fitted = True
        
    def predict(self, X):
        if not self.fitted:
            raise RuntimeError('please fit first')
        if self.fit_intercept:
            X = add_constant(X, prepend=False)
        if self.standardize:
            X[:, self.wf] = self.x_scaler.transform(X[:, self.wf])
        y_pred = X @ self.w.reshape(-1,1)
        return y_pred


class CustomENetCV:
    
    """Cross-validation of ElasticNet with customized penalties of covariates. 
    Most paramters are the same as sklearn's ElasticNetCV. Parameters listed 
    here are either new or need to be checked.
    
    Parameters
    ----------
    cv: 
        sklearn's CV splitter or an iterable yielding train and test indices.
    l1_ratio: 
        0 for Ridge, (0, 1) for ElasticNet, 1 for Lasso. Avoid (0, 0.01) if 
        possible. Currently only one value is allowed.
    alphas:
        List of alphas to try. For Ridge this parameter is mandatory. Also note
        that for Ridge, this alpha does not have the same magnitude as the 
        alpha of sklearn's Ridge due to the difference of objective functions. 
        Actually, alpha(sklearn's Ridge) = n_samples * alpha(here)
    metric:
        Function with y_true, y_pred as input and outputs the prediction 
        error. 
    standardize, fit_intercept:
        Same as CustomENet
    refit:
        Whether to refit with the whole dataset and the best alpha found.
    refit_*:
        Argument for refit. If None, it is set to the corresponding parameter 
        used in CV.
    verbose:
        Whether to print CV progress
    
    Attributes
    ----------
    cv_errs: 
        CV errors of path. Array of shape (len(alphas), n_CV).
    cv_min_err:
        Minimum CV error along the path.
    alpha_best:
        Alpha correspondng to to the minimum CV error.
    w:
        Only for refit=True. Regression coefficients. If fit_intercept=True, 
        w[-1] is the intercept.
    model_best:
        Only for refit=True. Sklearn's model used in the algorithm. Note this 
        is an intermediate model and should not be used for prediction.
    """
    
    def __init__(self, cv, l1_ratio: float = 0.5, 
                 eps=0.001, n_alphas=100, alphas=None, 
                 metric: callable = mean_squared_error, 
                 standardize = True,
                 fit_intercept = True,
                 positive = False,
                 tol=1e-4, max_iter=2000,
                 selection = 'cyclic',
                 refit = True,
                 refit_tol = None,
                 refit_max_iter = None, 
                 refit_selection = None,
                 random_state = None,
                 verbose = True
                 ):
        if l1_ratio == 0 and alphas is None:
            raise ValueError('for Ridge regression alphas cannot be None')
        self.cv = cv
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.metric = metric
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.tol = tol
        self.max_iter = max_iter
        self.selection = selection
        self.refit = refit
        if refit:
            if not refit_tol:
                self.refit_tol = tol
            if not refit_max_iter:
                self.refit_max_iter = max_iter
            if not refit_selection:
                self.refit_selection = selection
        self.random_state = random_state
        self.verbose = verbose
        self.fitted = False

    def prepare_en_(self, X, y, s):
        nsample = X.shape[0]
        sf_1 = ~(np.isinf(s) | (s == .0))
        sf_2 = s == .0
        X_1 = X[:, sf_1]
        s_1 = s[sf_1]
        X_2 = X[:, sf_2]
        X_1p = X_1 / s_1.reshape(1, -1)
        X_2i = np.linalg.pinv(X_2)
        H_2 = X_2 @ X_2i
        Q_2 = (np.identity(nsample) - H_2)
        y_q = np.asfortranarray(Q_2 @ y)
        X_1q = np.asfortranarray(Q_2 @ X_1p)
        return X_1p, X_1q, X_2i, s_1, y_q 
        
    def preprocess_(self, X: np.ndarray, 
                    y: np.ndarray, 
                    s = Union[None, np.ndarray]):
        X = X.copy()
        y = y.copy().reshape(-1, 1)
        if s is None:
            s = np.ones(X.shape[1])
        else:
            s = s.copy().ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError('y should be a 1-D vector with length equal to X.shape[0]')
        if X.shape[1] != s.shape[0]:
            raise ValueError('s should be a 1-D vector with length equal to X.shape[1]')
        if not np.all(s >= 0):
            raise ValueError('all elements in s must be non-negative')
        if self.fit_intercept:
            s = np.append(s, .0)
            X = add_constant(X, prepend=False)
        wf = ~(np.isinf(s) | (s == .0))
        x_scaler = StandardScaler()
        cv_dataset = []
        if hasattr(self.cv, '__iter__'):
            cv = self.cv
        else:
            cv = self.cv.split(X)
        for idx_tr, idx_test in cv:
            X_tr = X[idx_tr].copy()
            X_ts = X[idx_test].copy()
            y_tr = y[idx_tr].copy()
            y_ts = y[idx_test].copy()
            if self.standardize:
                X_tr[:, wf] = x_scaler.fit_transform(X_tr[:, wf])
                X_ts[:, wf] = x_scaler.transform(X_ts[:, wf])
            cv_dataset.append((X_tr, X_ts, y_tr, y_ts))
        X_s = X
        if self.standardize:
            X_s[:, wf] = x_scaler.fit_transform(X[:, wf])
            self.x_scaler = x_scaler
        else:
            self.x_scaler = None
        self.cv_dataset = cv_dataset
        self.wf = wf
        self.s = s
        self.nvar = X.shape[1]
        X_1p, X_1q, X_2i, s_1, y_q = self.prepare_en_(X_s, y, s)
        if self.alphas is None and self.l1_ratio != 0:
            self.alphas = alpha_grid(X_1q, y_q, 
                                     l1_ratio=self.l1_ratio, fit_intercept=False, 
                                     eps=self.eps, n_alphas=self.n_alphas)
        if self.refit:
            self.X_s, self.y = X_s, y
            self.X_1p, self.X_1q, self.X_2i, self.s_1, self.y_q = X_1p, X_1q, X_2i, s_1, y_q
    
    def se_path_(self, 
                X_tr, X_ts, y_tr, y_ts, 
                metric = mean_squared_error):
        X_1p, X_1q, X_2i, s_1, y_q = self.prepare_en_(X_tr, y_tr, self.s)
        # if self.l1_ratio != 0:
        #     gram = np.asfortranarray(X_1q.T @ X_1q)
        #     Xy = np.asfortranarray(X_1q.T @ y_q)
            
        #     model = ElasticNet(l1_ratio=self.l1_ratio, fit_intercept=False, precompute=gram, 
        #                   max_iter=self.max_iter, tol=self.tol, positive = self.positive,
        #                   warm_start=True, selection=self.selection, random_state = self.random_state)
        #     _, beta_1p, *_ = model.path(X_1q, y_q, l1_ratio=self.l1_ratio, 
        #                                alphas=self.alphas, precompute=gram, 
        #                                Xy=Xy, positive=self.positive, check_input=False)
        #     beta_1p = np.squeeze(beta_1p, axis=0)
        if self.l1_ratio != 0:
            gram = X_1q.T @ X_1q
            
            model = ElasticNet(l1_ratio=self.l1_ratio, fit_intercept=False, precompute=gram, 
                          max_iter=self.max_iter, tol=self.tol, positive = self.positive,
                          warm_start=True, selection=self.selection, random_state = self.random_state)
            beta_1p = []
            for alpha in self.alphas:
                model.set_params(alpha=alpha)
                model.fit(X_1q, y_q)
                beta_1p.append(model.coef_.copy())
            beta_1p = np.stack(beta_1p, axis=1)
        else:
            nsample = X_tr.shape[0]
            model = Ridge(fit_intercept=False, 
                          max_iter=self.max_iter, tol=self.tol, 
                          positive = self.positive,
                          random_state=self.random_state)
            beta_1p = []
            for alpha in self.alphas:
                model.set_params(alpha=nsample*alpha)
                model.fit(X_1q, y_q)
                beta_1p.append(model.coef_.copy().ravel())
            beta_1p = np.stack(beta_1p, axis=1)
        beta_1 = beta_1p / s_1.reshape(-1,1)
        beta_2 = X_2i @ (y_tr - X_1p @ beta_1p)
        sf_1 = self.wf
        sf_2 = self.s == .0
        w = np.zeros((self.nvar, len(self.alphas)))
        w[sf_1] = beta_1
        w[sf_2] = beta_2
        y_preds = X_ts @ w
        pred_errs = []
        for yp in y_preds.T:
            err = metric(y_ts, yp.reshape(-1,1))
            pred_errs.append(err)
        return pred_errs
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray, 
            s = Union[None, np.ndarray] 
            ):
        """Execute CV with customized penalties of covariates and possibly 
        refit the model with the best alpha found.
        
        Parameters
        ----------
        X : 
            Array with shape (n_samples, n_features).
        y : 
            Array with shape (n_samples,) or (n_samples, 1). Currently 
            multitarget is not suppoerted.
        s: 
            Penalty weight vector with shape (n_features,) or (n_features, 1).
            Weight must be non-negative but can be infinite.
        
        """
        
        self.preprocess_(X, y, s)
        cv_errs = []
        for cv_i, (X_tr, X_ts, y_tr, y_ts) in enumerate(self.cv_dataset):
            if self.verbose:
                print(f'CV round {cv_i+1}')
            errs = self.se_path_(X_tr, X_ts, y_tr, y_ts, 
                                 metric = self.metric)
            cv_errs.append(errs)
        cv_errs = np.stack(cv_errs, axis=1)
        self.cv_errs = cv_errs
        cv_errs_mean = cv_errs.mean(axis=1)
        err_min_idx = np.argmin(cv_errs_mean)
        self.cv_min_err = cv_errs_mean[err_min_idx]
        self.alpha_best = self.alphas[err_min_idx]
        if self.refit:
            if self.l1_ratio != 0:
                self.model_best = ElasticNet(alpha=self.alpha_best,
                               l1_ratio=self.l1_ratio, fit_intercept=False, 
                               max_iter=self.refit_max_iter, tol=self.refit_tol, 
                               positive = self.positive, selection=self.refit_selection,
                               random_state = self.random_state)
            else:
                nsample = X.shape[0]
                self.model_best = Ridge(alpha=nsample*self.alpha_best, 
                               fit_intercept=False, max_iter=self.refit_max_iter, 
                               tol=self.refit_tol, positive=self.positive, 
                               random_state=self.random_state)
            self.model_best.fit(self.X_1q, self.y_q)
            beta_1p = self.model_best.coef_.ravel()
            beta_1 = beta_1p / self.s_1
            beta_2 = (self.X_2i @ (self.y - self.X_1p @ beta_1p.reshape(-1,1))).ravel()
            sf_1 = self.wf
            sf_2 = self.s == .0
            w = np.zeros(self.nvar)
            w[sf_1] = beta_1
            w[sf_2] = beta_2
            self.w = w
            self.fitted = True
            
    def predict(self, X):
        if not self.fitted:
            if not self.refit:
                raise RuntimeError('refit is disabled. set refit=True and fit to enable predict')
            else:
                raise RuntimeError('please fit first')
        if self.fit_intercept:
            X = add_constant(X, prepend=False)
        if self.standardize:
            X[:, self.wf] = self.x_scaler.transform(X[:, self.wf])
        y_pred = X @ self.w.reshape(-1,1)
        return y_pred
