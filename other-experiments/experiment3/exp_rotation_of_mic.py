import os

from model_testing_interface_1d import Linear, SVM, Convolution
from data_utils_1d import read_data, k_fold_iter
from data_utils import read_data as read_data_2d

import numpy as np

models = {
    "Linear": Linear(),
    #"SVM": SVM(),
    "1-Channel CNN": Convolution(1),
    "4-Channel CNN": Convolution(4)
}

filename = "./results_exp_rotate_mic.csv"

path_no_rotation = "./mic_base_pos/data"
path_rotation = "./mic_rotated/data"

X, y = read_data_2d()# normalize=True)
X_rot1, y_rot1 = read_data(path_no_rotation, label_file=f"{path_no_rotation}/samples.csv")#, normalize=True)
X_rot2, y_rot2 = read_data(path_rotation, label_file=f"{path_rotation}/samples.csv")#, normalize=True)
results = []
k_fold = 5
seed = 42

use_indizes, = np.nonzero(np.abs(y[:,2] + 1.5) < 0.3)
X = X[use_indizes, :, :]
y = y[use_indizes, :2]

print("Beginning Evalution\n")

if not os.path.exists(filename):
    with open(filename, "x") as f:
        f.write("Modelname,Rotated,True,Predicted\n")

for X_train, y_train, X_test, y_test in k_fold_iter(X, y, k_fold, seed, val_set_size=0):
    for modelname, model in models.items():
        print(f' --- Training Model "{modelname}" ---')
        model.train(X_train, y_train, X_test, y_test)
        print(" --- Evaluating Model ---")
        pred1 = model.predict(X_rot1)
        pred2 = model.predict(X_rot2)
        with open(filename, "a") as f:
            for i in range(len(pred1)):
                f.write(f"{modelname},{0},{y_rot1[i,1]},{pred1[i]}\n")
                f.write(f"{modelname},{1},{y_rot2[i,1]},{pred2[i]}\n")
    break