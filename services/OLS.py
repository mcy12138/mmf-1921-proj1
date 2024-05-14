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
    #Construct the Data matrix and 
    T = factRet.shape[0]
    p = factRet.shape[1]
    one = np.ones((T, 1))
    X = np.hstack([one, factRet])
    B = np.linalg.inv(X.T@X)@X.T@returns
    
    alpha = B.iloc[0,]
    beta = B.iloc[1:,]
    print("Shape of X:", factRet.T.shape)
    print("Shape of B:", beta.T.shape)
    mu = alpha + np.dot(beta.T, gmean(factRet.T+1, axis = 1)-1)
    error = returns - np.dot(X, B)
    D = np.diag(np.linalg.norm(error, ord = 2, axis=0)/(T-p-1))
    
    F = np.cov(factRet.T)
    Q = np.dot(beta.T @ F, beta)+D
    
    mu = pd.DataFrame(mu)
    
    #Calculate the adjusted r square
    SS_Residual = np.sum(np.square(error))
    SS_Total = np.sum((returns - np.mean(returns, axis=0))**2)
    r_squared = 1 - (SS_Residual / SS_Total)
    adjusted_r_squared = np.mean(1 - ((1 - r_squared) * (T - 1) / (T - p - 1)))
    
    # Output results
    mu = pd.DataFrame(mu)
    
    
    

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------

    # mu =          % n x 1 vector of asset exp. returns
    # Q  =          % n x n asset covariance matrix
    # ----------------------------------------------------------------------

    return mu, Q, adjusted_r_squared

if __name__ == '__main__':
    adjClose = pd.read_csv("../MMF1921_AssetPrices.csv", index_col=0)
    factorRet = pd.read_csv("../MMF1921_FactorReturns.csv", index_col=0)
    
    adjClose.index = pd.to_datetime(adjClose.index)
    factorRet.index = pd.to_datetime(factorRet.index)
    
    #rf and factor returns
    riskFree = factorRet['RF']
    factorRet = factorRet.loc[:,factorRet.columns != 'RF'];
    
    #Identify the tickers and the dates
    tickers = adjClose.columns
    dates   = factorRet.index
    
    # Calculate the stocks monthly excess returns
    # pct change and drop the first null observation
    returns = adjClose.pct_change(1).iloc[1:, :]
    
    returns = returns  - np.diag(riskFree.values) @ np.ones_like(returns.values)
    # Align the price table to the asset and factor returns tables by discarding the first observation.
    adjClose = adjClose.iloc[1:,:]
    
    assert adjClose.index[0] == returns.index[0]
    assert adjClose.index[0] == factorRet.index[0]
    
    mu, Q= OLS(returns, factorRet, 0, 0)

    # result += alpha
    
    
    
    
    
        