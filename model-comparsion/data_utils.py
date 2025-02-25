# This file implements some helper functions to deal with data i/o and
# to perform the train-test splits, for data to predict two joint angles.

import os
import numpy as np
import scipy.io.wavfile
import pandas as pd


def get_labels(path):
    """Read the sample metadata."""
    return pd.read_csv(path, dtype={"notes":"str"})

def read_data(path="./data", 
            inputlength_s=4, 
            offset_s = 1,
            sample_rate=16000, 
            outputlength_samples=2048, 
            normalize=False,
            entangle_channels=True,
            apply_fft=True,
            label_file="./data/samples.csv"):
    """Reads in data and preprocesses it, by splitting it into chunks and applying fft.
    
    Parameters:
    path: Directory conaining the recordings
    inputlength_s > 0: The length of audiodata used from each sample (in s).
    offset_s: how much is cut off from the beginning (in s).
    sample_rate: Samplerate shared by each sample
    outputlength_samples: Length of one datapoint in samples
    normalize: Whether the audiodata is normalized to lie within [-1,1]
    entangle_channels: Whether the audio of channel 1 is added to the others
    apply_fft: Whether the audio is transformed to a (real) spectrum using fft. 
            Note that setting this to true results in each datapoint having the 
            dimension (outputlength_samples)/2+1
    label_file: Path of the sample metadata csv

    Output:
    X: Array of dim (samples, length, channels)
    y: Array of dimension (samples, 3) containing index, q_0 and q_3 for each sample
    """
    
    labels = get_labels(label_file)

    split_into = (inputlength_s * sample_rate // outputlength_samples)
    output_datapoints = (np.max(labels.idx)+1) * split_into
    req_inputlength = split_into * outputlength_samples

    print(f"Splitting each input into {split_into} datapoints, resulting in {output_datapoints} samples")

    X1 = np.zeros((output_datapoints, outputlength_samples))
    X2 = np.zeros((output_datapoints, outputlength_samples))
    X3 = np.zeros((output_datapoints, outputlength_samples))
    X4 = np.zeros((output_datapoints, outputlength_samples))

    y = np.repeat(np.hstack([np.arange(len(labels))[:,None], labels[["q_0", "q_3"]].to_numpy()]), split_into, axis=0)

    for idx, row in labels.iterrows():

        sr, data = scipy.io.wavfile.read(f"{path}/{row.idx}.wav")
        if sr != sample_rate:
            raise(f"Sample rate of {row.idx}.wav is {sr} instead of {sample_rate}")
        
        if len(data) < req_inputlength:
            raise(f"File {row.idx}.wav is not long enough")

        # Centering the data block within the available data
        start_of_block = int(sample_rate * offset_s)
        data_block1 = data[start_of_block:start_of_block+req_inputlength, 1]
        data_block2 = data[start_of_block:start_of_block+req_inputlength, 2]
        data_block3 = data[start_of_block:start_of_block+req_inputlength, 3]
        data_block4 = data[start_of_block:start_of_block+req_inputlength, 4]
        X1[idx*split_into:(idx+1)*split_into, :] = data_block1.reshape((split_into, outputlength_samples))
        if entangle_channels:
            X2[idx*split_into:(idx+1)*split_into, :] = 0.5*(data_block1 + data_block2).reshape((split_into, outputlength_samples))
            X3[idx*split_into:(idx+1)*split_into, :] = 0.5*(data_block1 + data_block3).reshape((split_into, outputlength_samples))
            X4[idx*split_into:(idx+1)*split_into, :] = 0.5*(data_block1 + data_block4).reshape((split_into, outputlength_samples))
        else:
            X2[idx*split_into:(idx+1)*split_into, :] = data_block2.reshape((split_into, outputlength_samples))
            X3[idx*split_into:(idx+1)*split_into, :] = data_block3.reshape((split_into, outputlength_samples))
            X4[idx*split_into:(idx+1)*split_into, :] = data_block4.reshape((split_into, outputlength_samples))

    if normalize:
        X1 = X1 / np.max(np.abs(X1))
        X2 = X2 / np.max(np.abs(X2))
        X3 = X3 / np.max(np.abs(X3))
        X4 = X4 / np.max(np.abs(X4))

    if apply_fft:
        print("Applying FFT")
        X1 = np.abs(np.fft.rfft(X1))
        X2 = np.abs(np.fft.rfft(X2))
        X3 = np.abs(np.fft.rfft(X3))
        X4 = np.abs(np.fft.rfft(X4))
    
    return np.stack([X1,X2,X3,X4], axis=-1), y



def k_fold_split(X, y, k_fold=5, seed=0):
    """Splits the data into k sets of the same size, while
    making sure that data belonging to the same pose ends up
    in the same set (to avoid mixing training and test data).

    Parameters:
    X: Data of dimension (samples, length, channels)
    y: labels of dimension (samples, 3)
    k_fold: into how many sets the data is split
    seed: seed used for shuffling the indizes

    Output:
    X_split: List of k-flod many arrays containing data from X
    y_split: List of k-fold many arrays containing labels from y
    """
    rng = np.random.default_rng(seed)
    shuffeled_idxs = np.unique(y[:,0])
    rng.shuffle(shuffeled_idxs)
    block_length = len(shuffeled_idxs)//k_fold
    X_split = []
    y_split = []
    for i in range(k_fold):
        split_idxs, = np.nonzero(np.isin(y[:,0], shuffeled_idxs[i*block_length:(i+1)*block_length]))
        X_split.append(X[split_idxs,:,:])
        y_split.append(y[split_idxs,1:])

    return X_split, y_split


def k_fold_iter(X, y, k_fold=5, seed=0, val_set_size=0):
    """Provides an iterator going through the data which
    returns a train, test and if needed validation set.

    Parameters:
    X: Data of dimension (samples, length, channels)
    y: labels of dimension (samples, 3)
    k_fold: into how many sets the data is split
    seed: seed used for shuffling the indizes
    val_set_size: Size of the validation set, if 0 no set is used

    If val_set_size > 0 this provides an iterator returning
    (X_train, y_train, X_val, y_val, X_test, y_test) and
    otherwise (X_train, y_train, X_test, y_test) where
    X_test, y_test always consist of one block, X_val, y_val
    consists of val_set_size blocks and X_train, y_train consists
    of k_fold - val_set_size - 1 blocks.
    """
    X_split, y_split = k_fold_split(X, y, k_fold, seed)
    for i in range(k_fold-val_set_size):
        train_X = np.concatenate(X_split[:i]+X_split[i+val_set_size+1:], axis=0)
        train_y = np.concatenate(y_split[:i]+y_split[i+val_set_size+1:], axis=0)
        if val_set_size > 0:
            test_X = np.concatenate(X_split[i:i+val_set_size], axis=0)
            test_y = np.concatenate(y_split[i:i+val_set_size], axis=0)
            yield train_X, train_y, test_X, test_y, X_split[i+val_set_size], y_split[i+val_set_size]
        else:
            yield train_X, train_y, X_split[i], y_split[i]
    for i in range(val_set_size):
        train_X = np.concatenate(X_split[i+1:k_fold-val_set_size+i], axis=0)
        train_y = np.concatenate(y_split[i+1:k_fold-val_set_size+i], axis=0)
        test_X = np.concatenate(X_split[:i]+X_split[k_fold-val_set_size+i:], axis=0)
        test_y = np.concatenate(y_split[:i]+y_split[k_fold-val_set_size+i:], axis=0)
        yield train_X, train_y, test_X, test_y, X_split[i], y_split[i]


def create_outputfile(filename):
    """Creates file for output if it doesn't exist."""
    Header = "iteration,model_id,true_q0,true_q3,pred_q0,pred_q3\n"
    if not os.path.exists(filename):
        with open(filename, "x") as f:
            f.write(Header)
    else:
        with open(filename, "r") as f:
            fileheader = f.readline()
        if fileheader != Header:
            raise("File exists but doesn't conform to standard.")
        

def write_output(
        filename,
        iteration,
        modelid,
        true_y,
        pred_y
):
    """Writes output and metadata to file."""
    results_to_write = ""
    for j in range(pred_y.shape[0]):
        results_to_write += (
            f"{iteration},{modelid}," +
            f"{true_y[j,0]:.8g}," + 
            f"{true_y[j,1]:.8g}," +
            f"{pred_y[j,0]:.8g}," +
            f"{pred_y[j,1]:.8g}\n"
        )
    with open(filename, "a") as f:
        f.write(results_to_write)