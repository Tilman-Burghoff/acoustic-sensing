import numpy as np
import scipy.io.wavfile
import os
import pandas as pd


def get_labels(path):
    return pd.read_csv(path, dtype={"notes":"str"})

def read_data(path="./data", 
            inputlength_s=5, 
            sample_rate=16000, 
            outputlength_samples=2048, 
            apply_fft=True,
            label_file="./data/samples.csv",
            label_column="q_0"):
    
    labels = get_labels(label_file)

    inputfiles = os.listdir(path)

    split_into = (inputlength_s * sample_rate // outputlength_samples)
    output_datapoints = len(inputfiles) * split_into
    req_inputlength = split_into * outputlength_samples

    # X_long = np.zeros((0,80896))

    print(f"Splitting each input into {split_into} datapoints, resulting in {output_datapoints} samples")

    X1 = np.zeros((output_datapoints, outputlength_samples))
    X2 = np.zeros((output_datapoints, outputlength_samples))
    X3 = np.zeros((output_datapoints, outputlength_samples))
    X4 = np.zeros((output_datapoints, outputlength_samples))
    y = np.full((output_datapoints, 2), np.nan)

    for i, file in enumerate(inputfiles):
        if file.split(".")[1] != "wav":
            continue
        sr, data = scipy.io.wavfile.read(path + "/" + file)
        idx = int(file.split(".")[0])
        if sr != sample_rate:
            raise(f"Samplerate of {file} is {sr} instead of {sample_rate}")
        
        if len(data) < req_inputlength:
            raise(f"File {file} is not long enough")

        # X_long = np.vstack((X_long, data[:,0]))
        start_of_block = (len(data) - req_inputlength) // 2
        data_block1 = data[start_of_block:start_of_block+req_inputlength, 1]
        data_block2 = data[start_of_block:start_of_block+req_inputlength, 2]
        data_block3 = data[start_of_block:start_of_block+req_inputlength, 1]
        data_block4 = data[start_of_block:start_of_block+req_inputlength, 2]
        X1[i*split_into:(i+1)*split_into, :] = data_block1.reshape((split_into, outputlength_samples))
        X2[i*split_into:(i+1)*split_into, :] = 0.5*(data_block1 + data_block2).reshape((split_into, outputlength_samples))
        X3[i*split_into:(i+1)*split_into, :] = 0.5*(data_block1 + data_block3).reshape((split_into, outputlength_samples))
        X4[i*split_into:(i+1)*split_into, :] = 0.5*(data_block1 + data_block4).reshape((split_into, outputlength_samples))
        y[i*split_into:(i+1)*split_into, :] = np.array((idx, labels.iloc[idx][label_column]))

    if apply_fft:
        print("applying FFT")
        X1 = np.abs(np.fft.rfft(X1))
        X2 = np.abs(np.fft.rfft(X2))
        X3 = np.abs(np.fft.rfft(X3))
        X4 = np.abs(np.fft.rfft(X4))
    
    return np.stack([X1,X2,X3,X4], axis=-1), y



def k_fold_split(X, y, k_fold=5, seed=0):
    rng = np.random.default_rng(seed)
    shuffeled_idxs = np.unique(y[:,0])[:-1] # remove nan
    rng.shuffle(shuffeled_idxs)
    block_length = len(shuffeled_idxs)//k_fold
    X_split = []
    y_split = []
    for i in range(k_fold):
        split_idxs, = np.nonzero(np.isin(y[:,0], shuffeled_idxs[i*block_length:(i+1)*block_length]))
        X_split.append(X[split_idxs,:,:])
        y_split.append(y[split_idxs,1])

    return X_split, y_split


def k_fold_iter(X, y, k_fold=5, seed=0):
    X_split, y_split = k_fold_split(X, y, k_fold, seed)
    for i in range(k_fold):
        train_X = np.concatenate(X_split[:i]+X_split[i+1:], axis=0)
        train_y = np.concatenate(y_split[:i]+y_split[i+1:], axis=0)
        yield train_X, train_y, X_split[i], y_split[i]


