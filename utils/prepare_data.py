import numpy as np


def prepare_data(data):
    """
    Перемешивает данные и делит их на обучающую и тестовую выборки
    """
    # Размеры данных
    m_all = data.shape[0]
    m_training = int(np.round(m_all * 0.7))  # 70% на обучение
    m_test = m_all - m_training

    # Генерация случайных индексов для перемешивания данных
    random_indices = np.random.permutation(m_all)

    # Перемешивание данных
    shuffled_data = data[random_indices]

    # Разделение перемешанных данных на обучающую и тестовую выборки
    X_train = shuffled_data[:m_training, :2]
    y_train = shuffled_data[:m_training, 2]

    X_test = shuffled_data[m_training:, :2]
    y_test = shuffled_data[m_training:, 2]

    return m_all, m_training, m_test, X_train, y_train, X_test, y_test
