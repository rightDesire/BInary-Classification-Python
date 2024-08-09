import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from cost_function_reg import cost_function_reg


def reg_analysis(X_train, y_train, X_test, y_test):
    """
    Возвращает массивы значений СКО при разных lambda
    """

    lambda_values = np.arange(0, 2.01, 0.01)

    J_train = np.zeros(len(lambda_values))
    J_test = np.zeros(len(lambda_values))

    for i, lambda_ in enumerate(lambda_values):
        initial_theta = np.zeros(X_train.shape[1])

        res = minimize(fun=cost_function_reg,
                       x0=initial_theta,
                       args=(X_train, y_train, lambda_),
                       method='BFGS',
                       jac=True,
                       options={'maxiter': 400})

        theta = res.x

        J_train[i], _ = cost_function_reg(theta, X_train, y_train, lambda_)
        J_test[i], _ = cost_function_reg(theta, X_test, y_test, lambda_)

    return lambda_values, J_train, J_test
