import numpy as np


def read_data(file_path):
    """
    Чтение данных из текстового файла
    """
    data = np.loadtxt(file_path, delimiter=',')
    return data
