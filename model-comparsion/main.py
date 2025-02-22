# This file contains the code to compare different models on the recorded training
# data in a k-fold train-test-split. model_testing_interface.py provides the models
# in an unified interface (train() and test()). data_utils provides helper functions 
# to read the recorded data and perform a train-test split.
# This implementation writes the raw resuöts into a csv-file

import os

from model_testing_interface import KNN, Linear, FullyConnected, Convolution
from data_utils import read_data, k_fold_iter

# Models selected for comparison
models = {
    0: KNN(10),
    1: Linear(),
    2: FullyConnected(3),
    3: FullyConnected(4),
    4: FullyConnected(5),
    5: Convolution(1),
    6: Convolution(2),
    7: Convolution(3),
    8: Convolution(4)
}

# output file, will be created if it doesn't exist, otherwise new data is appended
filename = "./results_raw_outside.csv" 

X, y = read_data()
results = []

# hyperparameters, the test set is only used for the neural networks to selcted the best epoch
k_fold = 10
test_set_size = 2
seed = 42

print("Beginning Evalution\n")

if not os.path.exists(filename):
    with open(filename, "x") as f:
        f.write("iteration,model_id,true_q0,true_q3,pred_q0,pred_q3\n")

for iteration, (X_train, y_train, X_test, y_test, X_val, y_val) in enumerate(k_fold_iter(X, y, k_fold, seed, test_set_size)):
    print(f"Evaluation round {iteration+1} of {k_fold}\n")
    for modelid, model in models.items():
        print(f' --- Training Model with id {modelid} ---')
        model.train(X_train, y_train, X_test, y_test)
        print(" --- Evaluating Model ---")
        preds = model.predict(X_val)

        # concat results to only do one expensive write operation
        results_to_write = ""
        for j in range(preds.shape[0]):
            results_to_write += (
                f"{iteration},{modelid}," +
                f"{y_val[j,0]:.8g}," + 
                f"{y_val[j,1]:.8g}," +
                f"{preds[j,0]:.8g}," +
                f"{preds[j,1]:.8g}\n"
            )
        with open(filename, "a") as f:
            f.write(results_to_write)
        print()