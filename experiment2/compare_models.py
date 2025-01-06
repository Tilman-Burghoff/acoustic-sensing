import numpy as np
import pandas as pd
from model_testing_interface import Linear, FullyConnected, Convolution
from model_evaluation_metrics import mean_square_error, R_squared, error_standard_deviation, avg_rec_std, outlier_ratio
from data_utils import read_data, k_fold_iter

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
    "Standard deviation per Recording (avg)": avg_rec_std,
    "Ratio of Outliers (>= 0.5 rad)": outlier_ratio
}


X, y = read_data()
results = []
k_fold = 5

print("Beginning Evalution\n")

for i, (X_train, y_train, X_test, y_test) in enumerate(k_fold_iter(X, y, k_fold)):
    print(f"Evaluation round {i} of {k_fold}\n")
    for modelname, model in models.items():
        print(f' --- Trainig Model "{modelname}" --- ')
        model.train(X_train, y_train)
        print(" --- Evaluating Model --- ")
        predictions = model.predict(X_test)
        for metricname, metric in metrics.items():
            results.append([i, modelname, metricname, metric(y_test, predictions)])
        print(" --- Evaluation Complete ---\n")

resultsdf = pd.DataFrame(results, columns=["Iteration", "Modelname", "Metricname", "Result"])
resultsdf.to_csv("./results.csv")