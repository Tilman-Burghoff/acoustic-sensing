import os

from model_testing_interface import KNN, Linear, SVM, FullyConnected, Convolution
from model_evaluation_metrics import mean_square_error, R_squared, error_standard_deviation, avg_rec_std, outlier_ratio
from data_utils import read_data, k_fold_iter

models = {
    "kNN": KNN(10),
    "Linear": Linear(),
    #"SVM": SVM(),
    "3-Layer FC": FullyConnected(3),
    "4-Layer FC": FullyConnected(4),
    "5-Layer FC": FullyConnected(5),
    "1-Channel CNN": Convolution(1),
    "2-Channel CNN": Convolution(2),
    "3-Channel CNN": Convolution(3),
    "4-Channel CNN": Convolution(4)
}

metrics = {
    "Mean Square Error": mean_square_error,
    "R squared": R_squared,
    "Standard Deviation of Error": error_standard_deviation,
    "Standard deviation per Recording (avg)": avg_rec_std,
    "Ratio of Outliers (>= 0.5 rad)": outlier_ratio
}

filename = "./results_contact_joint0.csv"

X, y = read_data()
results = []
k_fold = 10
test_set_size = 2
seed = 42

print("Beginning Evalution\n")

if not os.path.exists(filename):
    with open(filename, "x") as f:
        f.write("Iteration,Modelname,Metricname,Joint,Result\n")

for i, (X_train, y_train, X_test, y_test, X_val, y_val) in enumerate(k_fold_iter(X, y, k_fold, seed, test_set_size)):
    print(f"Evaluation round {i+1} of {k_fold}\n")
    for modelname, model in models.items():
        print(f' --- Training Model "{modelname}" ---')
        model.train(X_train, y_train, X_test, y_test)
        print(" --- Evaluating Model ---")
        predictions = model.predict(X_val)

        with open(filename, "a") as f:
            for metricname, metric in metrics.items():
                for joint in [0,1]: # we use joint 0 and 3, but htis makes indexing easier
                    res = metric(y_val[:,joint], predictions[:,joint])
                    print(f"{metricname} at joint {3*joint}: {res:.6f}")
                    f.write(f"{i},{modelname},{metricname},{joint*3},{res}\n")
        print()