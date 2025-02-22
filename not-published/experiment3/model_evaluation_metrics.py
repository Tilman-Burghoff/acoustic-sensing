import numpy as np

def mean_square_error(y_true, y_pred):
    return np.average((y_true - y_pred)**2)

def R_squared(y_true, y_pred):
    return 1 - np.sum((y_pred - y_true)**2) / np.sum((y_true - np.mean(y_true))**2)

def error_standard_deviation(y_true, y_pred):
    return np.std(y_true - y_pred)

def avg_rec_std(y_true, y_pred):
    recordings = np.unique(y_true)
    summed_std = 0
    for rec in recordings:
        indizes, = np.nonzero(y_true == rec)
        summed_std += np.std(y_pred[indizes])
    return summed_std / len(recordings)


def outlier_ratio(y_true, y_pred, count_as_outlier=0.5):
    return np.count_nonzero(np.abs(y_true - y_pred) >= count_as_outlier) / len(y_true)