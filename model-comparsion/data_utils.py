import numpy as np
import scipy.io.wavfile
import pandas as pd


def get_labels(path):
    return pd.read_csv(path, dtype={"notes":"str"})

def read_data(path="./data", 
            inputlength_s=4, 
            sample_rate=16000, 
            outputlength_samples=2048, 
            normalize=False,
            apply_fft=True,
            entangle_channels=True,
            label_file="./data/samples.csv"):
    
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
            raise(f"Samplerate of {row.idx}.wav is {sr} instead of {sample_rate}")
        
        if len(data) < req_inputlength:
            raise(f"File {row.idx}.wav is not long enough")

        # X_long = np.vstack((X_long, data[:,0]))
        start_of_block = 16000 # remove first second
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
        print("applying FFT")
        X1 = np.abs(np.fft.rfft(X1))
        X2 = np.abs(np.fft.rfft(X2))
        X3 = np.abs(np.fft.rfft(X3))
        X4 = np.abs(np.fft.rfft(X4))
    
    return np.stack([X1,X2,X3,X4], axis=-1), y



def k_fold_split(X, y, k_fold=5, seed=0):
    rng = np.random.default_rng(seed)
    shuffeled_idxs = np.unique(y[:,0])
    print(shuffeled_idxs[-1])
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



