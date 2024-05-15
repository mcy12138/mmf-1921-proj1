import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt


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

    return mu, Q, adjusted_r_squared



# def r2_plot(returns, factRet, label, K=None):
#     lambda_list = np.arange(0.01, 2, 0.1)
#     r2_list = []
#     for l in  lambda_list:
#         _, _, adj_r2 = LASSO(returns, factRet, l, K=None)
#         r2_list.append(adj_r2)
#
#     plt.plot(lambda_list, r2_list, label=label)
#     plt.title("R^2 by lambda value LASSO")
#     plt.xlabel('lambda value')
#     plt.ylabel('Adjusted R Square')
#     plt.legend()



if __name__ == '__main__':
    adjClose = pd.read_csv("../MMF1921_AssetPrices_1.csv", index_col=0)
    factorRet = pd.read_csv("../MMF1921_FactorReturns_1.csv", index_col=0)

    adjClose.index = pd.to_datetime(adjClose.index)
    factorRet.index = pd.to_datetime(factorRet.index)

    # Extract the risk-free rate and factor returns
    riskFree = factorRet['RF']
    factorRet = factorRet.loc[:, factorRet.columns != 'RF']

    # Calculate the stocks' monthly excess returns
    returns = adjClose.pct_change(1).iloc[1:, :]
    returns -= np.diag(riskFree.values) @ np.ones_like(returns.values)
    # r2_plot(returns, factorRet)

#     # Select a lambda value or use cross-validation or empirical testing to determine it
    lambda_val = 0.3  # Example value, adjust based on empirical testing or cross-validation

#     # Get LASSO results
    mu, Q, adjusted_r_squared = RIDGE(returns, factorRet, lambda_val, K=None)
    # print(mu)
    # print(Q)
    # print(adjusted_r_squared)