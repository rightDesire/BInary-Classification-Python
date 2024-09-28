import numpy as np


def shuffle_data(data):
    """
    Перемешивает данные.

    Parameters:
    data (numpy array): Исходные данные

    Returns:
    numpy array: Перемешанные данные
    """
    # Перемешивание данных
    random_indices = np.random.permutation(data.shape[0])
    shuffled_data = data[random_indices]
    return shuffled_data
