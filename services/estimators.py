import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats import gmean


def OLS(returns, factRet, lambda_, K):
    """
    % Use this function to perform an OLS regression. Note that you will
    % not use lambda or K in this model (lambda is for LASSO, and K is for
    % BSS).
    """
    # Construct the Data matrix and
    T = factRet.shape[0]
    p = factRet.shape[1]
    one = np.ones((T, 1))
    X = np.hstack([one, factRet])
    B = np.linalg.inv(X.T @ X) @ X.T @ returns

    alpha = B.iloc[0,]
    beta = B.iloc[1:, ]
    mu = alpha + np.dot(beta.T, gmean(factRet.T + 1, axis=1) - 1)
    error = returns - np.dot(X, B)
    D = np.diag(np.linalg.norm(error, ord=2, axis=0) / (T - p - 1))

    F = np.cov(factRet.T)
    Q = np.dot(beta.T @ F, beta) + D

    mu = pd.DataFrame(mu)

    # Calculate the adjusted r square
    SS_Residual = np.sum(np.square(error))
    SS_Total = np.sum((returns - np.mean(returns, axis=0)) ** 2)
    r_squared = 1 - (SS_Residual / SS_Total)
    adjusted_r_squared = np.mean(1 - ((1 - r_squared) * (T - 1) / (T - p - 1)))

    # Output results
    mu = pd.DataFrame(mu)

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------

    # mu =          % n x 1 vector of asset exp. returns
    # Q  =          % n x n asset covariance matrix
    # ----------------------------------------------------------------------

    return mu, Q


def FF(returns, factRet, lambda_, K):
    """
    % Use this function to calibrate the Fama-French 3-factor model. Note
    % that you will not use lambda or K in this model (lambda is for LASSO,
    % and K is for BSS).
    """
    # Select the features used fot ff model
    factRet = factRet[['Mkt_RF', 'SMB', 'HML']]
    T = factRet.shape[0]
    p = factRet.shape[1]
    one = np.ones((T, 1))
    X = np.hstack([one, factRet])
    B = np.linalg.inv(X.T @ X) @ X.T @ returns

    alpha = B.iloc[0,]
    beta = B.iloc[1:, ]
    mu = alpha + np.dot(beta.T, gmean(factRet.T + 1, axis=1) - 1)
    error = returns - np.dot(X, B)
    D = np.diag(np.linalg.norm(error, ord=2, axis=0) / (T - p - 1))
    # residual = returns - result_matrix.T
    # residual -= alpha

    F = np.cov(factRet.T)
    # # res = residual.cov()
    # res = 1/(T-p-1)
    # D = np.diagonal(res)
    Q = np.dot(beta.T @ F, beta) + D

    # Calculate the adjusted r square
    SS_Residual = np.sum(np.square(error))
    SS_Total = np.sum((returns - np.mean(returns, axis=0)) ** 2)
    r_squared = 1 - (SS_Residual / SS_Total)
    adjusted_r_squared = np.mean(1 - ((1 - r_squared) * (T - 1) / (T - p - 1)))
    # mu = gmean(result_matrix + 1, axis = 1)-1
    # mu = np.mean(result_matrix, axis = 1)
    mu = pd.DataFrame(mu)
    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------

    # mu =          % n x 1 vector of asset exp. returns
    # Q  =          % n x n asset covariance matrix
    # ----------------------------------------------------------------------

    return mu, Q


def LASSO(returns, factRet, lambda_val, K):
    """
    % Use this function for the LASSO model. Note that you will not use K
    % in this model (K is for BSS).
    %
    % You should use an optimizer to solve this problem. Be sure to comment
    % on your code to (briefly) explain your procedure.
    """

    # Number of assets (n) and number of factors (m)
    n_assets = returns.shape[1]
    n_factors = factRet.shape[1]

    # Create a variable for coefficients, including intercept as the first entry
    B = cp.Variable((n_factors + 1, n_assets))

    T = factRet.shape[0]
    X = np.hstack([np.ones((T, 1)), factRet.values])

    residuals = returns.values - X @ B

    # L2 norm of residuals (sum of squared residuals)
    l2_norm = cp.sum_squares(residuals)

    # L1 norm of the coefficients for regularization
    l1_norm = cp.norm(B, 1)

    objective = cp.Minimize(l2_norm + lambda_val * l1_norm)

    # Define the problem and solve
    problem = cp.Problem(objective)
    problem.solve()

    geometric_mean_factRet = gmean(factRet + 1, axis=0) - 1
    geometric_mean_X = np.hstack([1, geometric_mean_factRet])
    predicted_returns = geometric_mean_X @ B.value

    mu = pd.DataFrame(predicted_returns, index=returns.columns, columns=['Expected Return'])

    residuals_geom_mean = returns - (X @ B.value)
    # p =
    tol = 1e-4
    p = np.count_nonzero(np.abs(B.value) > tol, axis=0)
    # D = np.diag(np.linalg.norm(residuals_geom_mean, ord=2, axis=0) / (T - factRet.shape[1] - 1))
    D = np.diag(np.linalg.norm(residuals_geom_mean, ord=2, axis=0) / (T - p - 1))
    F = np.cov(factRet.T)
    Q = B.value[1:].T @ F @ B.value[1:] + D

    # Q = np.cov(residuals_geom_mean, rowvar=False)
    Q = pd.DataFrame(Q, index=returns.columns, columns=returns.columns)
    #Calculate the adjusted r square
    SS_Residual = np.sum(np.square(residuals_geom_mean), axis=0)
    SS_Total = np.sum(np.square(returns - np.mean(returns, axis=0)))
    r_squared = 1 - (SS_Residual / SS_Total)
    adjusted_r_squared = np.mean(1 - ((1 - r_squared) * (T - 1) / (T - p - 1)), axis=0)

    return mu, Q


def RIDGE(returns, factRet, lambda_val, K=None):
    """
    % Use this function for the LASSO model. Note that you will not use K
    % in this model (K is for BSS).
    %
    % You should use an optimizer to solve this problem. Be sure to comment
    % on your code to (briefly) explain your procedure.
    """

    # Number of assets (n) and number of factors (m)
    n_assets = returns.shape[1]
    n_factors = factRet.shape[1]

    # Create a variable for coefficients, including intercept as the first entry
    B = cp.Variable((n_factors + 1, n_assets))

    T = factRet.shape[0]
    X = np.hstack([np.ones((T, 1)), factRet.values])

    residuals = returns.values - X @ B

    # L2 norm of residuals (sum of squared residuals)
    l2_norm = cp.sum_squares(residuals)

    # L2 norm of the coefficients for regularization
    l2_regularization = cp.norm(B, 2)

    objective = cp.Minimize(l2_norm + lambda_val * l2_regularization)

    # Define the problem and solve
    problem = cp.Problem(objective)
    problem.solve()

    geometric_mean_factRet = gmean(factRet + 1, axis=0) - 1
    geometric_mean_X = np.hstack([1, geometric_mean_factRet])
    predicted_returns = geometric_mean_X @ B.value

    mu = pd.DataFrame(predicted_returns, index=returns.columns, columns=['Expected Return'])

    residuals_geom_mean = returns - (X @ B.value)
    tol = 1e-4
    p = np.count_nonzero(np.abs(B.value) > tol, axis=0)
    # D = np.diag(np.linalg.norm(residuals_geom_mean, ord=2, axis=0) / (T - factRet.shape[1] - 1))
    D = np.diag(np.linalg.norm(residuals_geom_mean, ord=2, axis=0) / (T - p - 1))
    F = np.cov(factRet.T)
    Q = B.value[1:].T @ F @ B.value[1:] + D

    # Q = np.cov(residuals_geom_mean, rowvar=False)
    Q = pd.DataFrame(Q, index=returns.columns, columns=returns.columns)
    #Calculate the adjusted r square
    SS_Residual = np.sum(np.square(residuals_geom_mean), axis=0)
    SS_Total = np.sum(np.square(returns - np.mean(returns, axis=0)))
    r_squared = 1 - (SS_Residual / SS_Total)
    adjusted_r_squared = np.mean(1 - ((1 - r_squared) * (T - 1) / (T - p - 1)), axis=0)

    return mu, Q
