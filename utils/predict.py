from sigmoid import sigmoid


def predict(theta, X):
    """
    Возвращает булево значение в зависимости от гипотезы
    """

    return sigmoid(X @ theta) >= 0.5
