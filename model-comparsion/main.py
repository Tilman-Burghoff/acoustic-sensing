# This file contains the code to compare different models on the recorded training
# data in a k-fold train-test-split. model_testing_interface.py provides the models
# in an unified interface (train() and test()). data_utils provides helper functions 
# to read the recorded data and perform a train-test split.
# This implementation writes the raw results into a csv-file

from model_testing_interface import KNN, Linear, FullyConnected, Convolution
from data_utils import read_data, create_outputfile, k_fold_iter, write_output


def write_results(results, filename):
    """Writes results to file."""


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

# if you don't use the file path as given by the data recording script,
# use read_data(path=[path_to_folder], label_file=[path_to_metadata])
X, y = read_data() 
results = []

# hyperparameters, the test set is only used for the neural networks to selcted the best epoch
k_fold = 10
test_set_size = 2
seed = 42

print("Beginning Evalution\n")

create_outputfile(filename)

for iteration, (X_train, y_train, X_test, y_test, X_val, y_val) in enumerate(k_fold_iter(X, y, k_fold, seed, test_set_size)):
    print(f"Evaluation round {iteration+1} of {k_fold}\n")
    for modelid, model in models.items():
        print(f' --- Training Model with id {modelid} ---')
        model.train(X_train, y_train, X_test, y_test)
        print(" --- Evaluating Model ---")
        preds = model.predict(X_val)

        # concat results to only do one expensive write operation
        write_output(filename, iteration, modelid, y_test, preds)
        print()