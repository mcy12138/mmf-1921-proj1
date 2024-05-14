from services.strategies import *


def project_function(periodReturns, periodFactRet, x0):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """
    Strategy = OLS_MVO()
    x = Strategy.execute_strategy(periodReturns, periodFactRet)
    return x
