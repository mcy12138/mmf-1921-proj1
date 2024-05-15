from services.strategies import *
import warnings
warnings.filterwarnings("ignore")


def project_function(periodReturns, periodFactRet, x0):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """
    Strategy = OLS_MVO()
    Strategy = OLS_CVaR()
    Strategy = OLS_RP()
    Strategy = OLS_robust()
    Strategy = ridge_MVO()
    Strategy = ridge_CVaR()
    Strategy = ridge_RP()
    Strategy = ridge_robust()
    Strategy = LASSO_MVO()
    Strategy = LASSO_CVaR()
    Strategy = LASSO_RP()
    Strategy = LASSO_robust()
    Strategy = FF_MVO()
    Strategy = FF_CVaR()
    Strategy = FF_RP()
    Strategy = FF_robust()
    Strategy = BSS_MVO()
    Strategy = BSS_CVaR()
    Strategy = BSS_RP()
    Strategy = BSS_robust()

    x = Strategy.execute_strategy(periodReturns, periodFactRet)
    return x
