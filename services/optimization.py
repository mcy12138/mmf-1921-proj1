import cvxpy as cp
import numpy as np
from scipy.stats import chi2
import warnings
warnings.filterwarnings("ignore")


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
    A = A.to_numpy()

    # Define and solve using CVXPY
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)),
                      [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb])
    prob.solve(verbose=False)
    return x.value

def RP(Q):
    n = Q.shape[0]

    # Assign an arbitrary value for kappa
    kappa = 5

    # Define the optimization variable
    x = cp.Variable(n)

    # Objective function
    objective = 0.5 * cp.quad_form(x, Q) - kappa * cp.sum(cp.log(x))

    # Constraints
    # constraints = [x >= 0, cp.sum(x) == 1]  # x >= 0 and sum(x) = 1
    constraints = [x >= 0]

    # Define and solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()

    # Recover the weights
    x_value = x.value/np.sum(x.value)

    # Return the optimized portfolio and the associated cost
    return x_value


def robustMVO(mu, Q, lambda_, alpha, T):
    # Number of assets
    n = Q.shape[0]

    # Radius of the uncertainty set
    ep = np.sqrt(chi2.ppf(alpha, n))

    # Theta (squared standard error of expected returns)
    Theta = np.diag(np.diag(Q)) / T

    # Square root of Theta
    sqrtTh = np.sqrt(Theta)

    # Define the variable for the optimization
    x = cp.Variable(n)

    A = mu.T.to_numpy()

    # Objective function
    obj = cp.Minimize(lambda_ * cp.quad_form(x, Q) - A @ x + ep * cp.norm(sqrtTh @ x, 2))

    # Constraints
    constraints = [
        cp.sum(x) == 1,
        x >= 0
    ]

    # Solve the problem
    prob = cp.Problem(obj, constraints)
    prob.solve()

    return x.value


def CVaR(mu, rets, alpha=0.95):
    # Find the total number of assets
    S, n = rets.shape

    R = np.mean(mu)

    # Define the variables for the optimization problem
    x = cp.Variable(n)
    z = cp.Variable(S)
    gamma = cp.Variable()
    lb = np.zeros(n)

    # Define the constraints
    constraints = [
        z >= 0,
        # z >= -rets @ x - gamma,
        z >= cp.matmul(-rets, x) - gamma,
        cp.sum(x) == 1,
        # mu.T @ x >= R,
        cp.matmul(mu.T, x) >= R,
        x >= lb
    ]

    # Define the objective function
    k = 1 / ((1 - alpha) * S)
    objective = cp.Minimize(gamma + k * cp.sum(z))

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return x.value

