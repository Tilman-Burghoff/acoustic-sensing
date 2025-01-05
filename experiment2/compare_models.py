import numpy as np
import pandas as pd
from model_testing_interface import Linear, FullyConnected, Convolution
from model_evaluation_metrics import mean_square_error, R_squared, error_standard_deviation, avg_rec_std

models = {
    "Linear": Linear(),
    "2-Layer FC": FullyConnected(0),
    "3-Layer FC": FullyConnected(1),
    "4-Layer FC": FullyConnected(2),
    "1-Channel CNN": Convolution(1),
    "2-Channel CNN": Convolution(2),
    "3-Channel CNN": Convolution(3),
    "4-Channel CNN": Convolution(4)
}

metrics = {
    "Mean Sqaure Error": mean_square_error,
    "R squared": R_squared,
    "Standard Deviation of Error": error_standard_deviation,
    "Standard deviation per Recording (avg)": avg_rec_std
}
