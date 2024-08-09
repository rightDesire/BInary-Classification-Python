import numpy as np


def map_feature(X1, X2, num_features):
    """
    Возвращает массив с добавлением полиномиальных признаков
    """

    # Единичный столбец, размерности вектора X1
    out = np.ones((X1.shape[0], 1))

    # Расчет максимальной степени полинома
    degree = 1
    while (degree + 1) * (degree + 2) / 2 <= num_features:
        degree += 1

    # Генерация полиномиальных признаков
    if degree > 1:
        for i in range(1, degree + 1):
            for j in range(i + 1):
                if out.shape[1] < num_features:
                    # Расчет полиномных признаков
                    feature = (X1 ** (i - j)) * (X2 ** j)
                    out = np.hstack((out, feature[:, np.newaxis]))
    else:
        # Если degree <= 1, просто возвращаем исходные X1 и X2
        out = np.hstack((out, X1[:, np.newaxis], X2[:, np.newaxis]))

    return out
