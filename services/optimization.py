import cvxpy as cp
import numpy as np
import pandas as pd


def MVO(mu, Q):
    """
    #---------------------------------------------------------------------- Use this function to construct an example of a MVO portfolio.
    #
    # An example of an MVO implementation is given below. You can use this
    # version of MVO if you like, but feel free to modify this code as much
    # as you need to. You can also change the inputs and outputs to suit
    # your needs.

    # You may use quadprog, Gurobi, or any other optimizer you are familiar
    # with. Just be sure to include comments in your code.

    # *************** WRITE YOUR CODE HERE ***************
    #----------------------------------------------------------------------
    """

    # Find the total number of assets
    n = len(mu)

    # Set the target as the average expected return of all assets
    targetRet = np.mean(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Add the expected return constraint
    A = -1 * mu.T
    b = -1 * targetRet

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Define and solve using CVXPY
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)),
                      [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb])
    prob.solve(verbose=False)
    return x.value

def CVaR(mu, rets, alpha=0.95):
    """
    #Use this function to construct an example of a CVaR portfolio.

    """

    # Find the total number of assets
    S, n = rets.shape

    # Define the target return (as an example, set it as 10% higher than the average asset return)
    R = 1.1 * np.mean(mu)

    # Define the variables for the optimization problem
    x = cp.Variable(n)
    z = cp.Variable(S)
    gamma = cp.Variable()
    lb = np.zeros(n)

    # Define the constraints
    constraints = [
        z >= 0,
        z >= -rets @ x - gamma,
        cp.sum(x) == 1,
        mu.T @ x >= R,
        x >= lb
    ]

    # Define the objective function
    k = 1 / ((1 - alpha) * S)
    objective = cp.Minimize(gamma + k * cp.sum(z))

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return x.value

