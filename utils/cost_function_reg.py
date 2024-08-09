import numpy as np
from sigmoid import sigmoid


def cost_function_reg(theta, X, y, lambda_):
    """
    Возвращает СКО и результат градиентного спуска при заданных параметрах
    """

    m = len(y)
    hypothesis = sigmoid(X @ theta)

    # Предотвращает логарифм от нуля
    epsilon = 1e-10
    J = (1 / m) * np.sum(-y * np.log(hypothesis + epsilon) - (1 - y) * np.log(1 - hypothesis + epsilon)) + \
        (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)

    grad = (1 / m) * (X.T @ (hypothesis - y))
    grad[1:] += (lambda_ / m) * theta[1:]

    return J, grad
