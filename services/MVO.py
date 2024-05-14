import cvxpy as cp
import numpy as np
import pandas as pd


def MVO(mu, Q, targetRet):
    """
    % Use this function to construct your MVO portfolio subject to the
    % target return, with short sales disallowed.
    %
    % You may use quadprog, Gurobi, cvxpy or any other optimizer you are familiar
    % with. Just be sure to comment on your code to (briefly) explain your
    % procedure.


    """

    # Find the total number of assets
    # n = len(mu)

    # *************** WRITE YOUR CODE HERE ***************
    # -----------------------------------------------------------------------

    # x =           % Optimal asset weights
    # ----------------------------------------------------------------------
    n = mu.shape[0]  # Number of assets
    w = cp.Variable(n)  # Portfolio weights

    mu = mu.values.flatten() if isinstance(mu, pd.DataFrame) else mu

    # Portfolio return as dot product of weights and returns
    ret = mu.T @ w

    # Portfolio variance as quadratic form
    risk = cp.quad_form(w, Q.values) if isinstance(Q, pd.DataFrame) else cp.quad_form(w, Q)

    # Objective: Minimize risk
    objective = cp.Minimize(risk)

    # Constraints
    constraints = [
        cp.sum(w) == 1,  # Sum of weights must be 1
        ret >= targetRet,  # Ensure target return is met or exceeded
        w >= 0  # no short selling
    ]

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Return the optimal weights as a NumPy array
    return w.value


# if __name__ == '__main__':
    
#     adjClose = pd.read_csv("../MMF1921_AssetPrices.csv", index_col=0)
#     factorRet = pd.read_csv("../MMF1921_FactorReturns.csv", index_col=0)

#     adjClose.index = pd.to_datetime(adjClose.index)
#     factorRet.index = pd.to_datetime(factorRet.index)

#     # Extract the risk-free rate and factor returns
#     riskFree = factorRet['RF']
#     factorRet = factorRet.loc[:, factorRet.columns != 'RF']

#     # Calculate the stocks' monthly excess returns
#     returns = adjClose.pct_change(1).iloc[1:, :]
#     returns -= np.diag(riskFree.values) @ np.ones_like(returns.values)

#     # Select a lambda value or use cross-validation or empirical testing to determine it
#     lambda_val = 0.3  # Example value, adjust based on empirical testing or cross-validation

#     # Use results from LASSO as a Test
#     mu, Q = LASSO(returns, factorRet, lambda_val, K=None)
#     w = MVO(mu, Q, targetRet=0.01)
#     print(w)

