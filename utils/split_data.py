import numpy as np


def split_data(shuffled_data, train_ratio=0.7):
    """
    Делит данные на обучающую и тестовую выборки.

    Parameters:
    shuffled_data (numpy array): Перемешанные данные
    train_ratio (float): Процент данных для обучения (по умолчанию 70%)

    Returns:
    X_train, y_train, X_test, y_test: Разделенные данные
    """
    # Размеры выборок
    m_all = shuffled_data.shape[0]
    m_training = int(np.round(m_all * train_ratio))  # 70% на обучение

    # Разделение перемешанных данных на обучающую и тестовую выборки
    X_train = shuffled_data[:m_training, :2]
    y_train = shuffled_data[:m_training, 2]

    X_test = shuffled_data[m_training:, :2]
    y_test = shuffled_data[m_training:, 2]

    return X_train, y_train, X_test, y_test
