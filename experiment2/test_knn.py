import numpy as np
import pandas as pd
from model_testing_interface import KNN
from model_evaluation_metrics import mean_square_error, R_squared, error_standard_deviation, avg_rec_std, outlier_ratio
from data_utils import read_data, k_fold_iter



models = {nbh:KNN(nbh) for nbh in range(1,25)}

metrics = {
    "Mean Square Error": mean_square_error,
    "R squared": R_squared,
    "Standard Deviation of Error": error_standard_deviation,
    "Standard deviation per Recording (avg)": avg_rec_std,
    "Ratio of Outliers (>= 0.5 rad)": outlier_ratio
}


X, y = read_data()
results = []
k_fold = 10
test_set_size = 2
seed = 42

print("Beginning Evalution\n")

for i, (X_train, y_train, X_test, y_test, X_val, y_val) in enumerate(k_fold_iter(X, y, k_fold, seed, test_set_size)):
    print(f"Evaluation round {i+1} of {k_fold}\n")
    for modelname, model in models.items():
        print(f' --- Training Model "{modelname}" ---')
        model.train(X_train, y_train, X_test, y_test)
        print(" --- Evaluating Model ---\n")
        predictions = model.predict(X_val)
        for metricname, metric in metrics.items():
            results.append([i, modelname, metricname, metric(y_val, predictions)])

resultsdf = pd.DataFrame(results, columns=["Iteration", "Modelname", "Metricname", "Result"])
resultsdf.to_csv("./results_KNN.csv")