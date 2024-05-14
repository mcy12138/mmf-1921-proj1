import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats import gmean


def BSS(returns, factRet, lambda_, K):

    def bss_single_asset(returns, factRet, lambda_, K):
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

        l2_norm = cp.sum_squares(residuals)

        objective = cp.Minimize(l2_norm)

        select = cp.Variable((n_factors + 1, n_assets), boolean=True)

        constraints = [
            cp.sum(select) <= K,
            B <= select * 20,
            B >= -select * 20
        ]

        # Define the problem and solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GUROBI)

        # Expected returns (mean of predicted returns)
        geometric_mean_factRet = gmean(factRet + 1, axis=0) - 1
        geometric_mean_X = np.hstack([1, geometric_mean_factRet])
        mu = geometric_mean_X @ B.value
        mu = pd.DataFrame(mu, columns=['Expected Return'], index=returns.columns)

        # Extract results
        predicted_returns = X @ B.value
        # Covariance matrix of residuals
        residuals = returns.values - predicted_returns
        residuals = pd.DataFrame(residuals)

        return mu, residuals, B.value

    mu_total = pd.DataFrame()
    residual_total = pd.DataFrame()
    B_total = []
    for i in range(20):
        returns_single = pd.DataFrame(returns.iloc[:, i])

        # Get LASSO results
        mu, res, B = bss_single_asset(returns_single, factRet, 0, K)
        mu_total = pd.concat([mu_total, mu], ignore_index=True)
        residual_total = pd.concat([residual_total, res], axis=1)
        B_total.append(B.reshape(-1, 1))

        # print(beta_matrix)
    D = np.diag(np.linalg.norm(residual_total, ord=2, axis=0) / (factRet.shape[0] - 4 - 1))
    F = np.cov(factRet.T)
    B_final = np.hstack(tuple(B_total))
    Q = B_final[1:].T @ F @ B_final[1:] + D
    Q = pd.DataFrame(Q)
    #Calculate the adjusted r square
    SS_Total = np.sum(np.square(returns - np.mean(returns)), axis=0)
    SS_Residual = np.sum(np.square(residual_total), axis=0)
    SS_Residual.index = SS_Total.index
    r_squared = 1 - (SS_Residual / SS_Total)
    adjusted_r_squared = np.mean(1 - ((1 - r_squared) * (factRet.shape[0] - 1) / (factRet.shape[0] - 4 - 1)), axis=0)

    return mu_total, Q, adjusted_r_squared


if __name__ == '__main__':
    adjClose = pd.read_csv("../MMF1921_AssetPrices.csv", index_col=0)
    factorRet = pd.read_csv("../MMF1921_FactorReturns.csv", index_col=0)

    adjClose.index = pd.to_datetime(adjClose.index)
    factorRet.index = pd.to_datetime(factorRet.index)

    # Extract the risk-free rate and factor returns
    riskFree = factorRet['RF']
    factorRet = factorRet.loc[:, factorRet.columns != 'RF']

    # Calculate the stocks' monthly excess returns
    returns = adjClose.pct_change(1).iloc[1:, :]
    returns -= np.diag(riskFree.values) @ np.ones_like(returns.values)

    # Get LASSO results
    mu, Q, r_squared = BSS(returns, factorRet, 0, 4)

    print("mu is: ", mu)
    print("Q is: ", Q)
    print("Adjusted R Squared", r_squared)

    # Calculate the stocks' monthly excess returns
