import numpy as np
import pandas as pd
import os

from model_testing_interface import KNN, Linear, SVM, FullyConnected, Convolution
from model_evaluation_metrics import mean_square_error, R_squared, error_standard_deviation, avg_rec_std, outlier_ratio
from data_utils import read_data, k_fold_iter

model = FullyConnected(3)
X, y = read_data()
results = []
k_fold = 10
test_set_size = 2
seed = 42

print("Beginning Evalution\n")


if not os.path.exists("./MSE_by_pos.csv"):
    with open("MSE_by_pos.csv", "x") as f:
        f.write("Iteration,q_0,q_3,joint,MSE\n")

for i, (X_train, y_train, X_test, y_test, X_val, y_val) in enumerate(k_fold_iter(X, y, k_fold, seed, test_set_size)):
    print(f"Evaluation round {i+1} of {k_fold}\n")
    
    print(f' --- Training Model ---')
    model.train(X_train, y_train, X_test, y_test)
    print(" --- Evaluating Model ---")
    predictions = model.predict(X_val)

    recordings = np.unique(y_val, axis=0)
    summed_mse = 0
    with open("MSE_by_pos.csv", "a") as f:
        for rec in recordings:
            indizes, _ = np.nonzero(y_val == rec)
            mse = np.mean((predictions[indizes, :] - rec[None,:])**2, axis=0) 
            f.write(f"{i},{rec[0]},{rec[1]},{0},{mse[0]}\n")
            f.write(f"{i},{rec[0]},{rec[1]},{3},{mse[1]}\n")
            summed_mse += mse[0] + mse[1]
    print(f"Validation loss: {summed_mse/(len(recordings)*2)}\n")