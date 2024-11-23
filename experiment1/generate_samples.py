import scipy.io.wavfile
import numpy as np
import os

inputpath = "./exp1_position_classification" # input("Enter data dictonary: ")
outputpath = "./exp1_position_classification"  # input("Enter output dictonary: ")
inputlength =  30 # int(input("Enter length of input data (in s): "))
SR = 48000 # int(input("Samplerate: "))
outputlength = int(input("Enter length of output data (in samples): "))
applyFFT = True if input("Apply FFT (y/N) ").lower() == "y" else False

inputfiles = os.listdir(inputpath)

split_into = (inputlength * SR // outputlength)
output_datapoints = len(inputfiles) * split_into
req_inputlength = split_into * outputlength

print(f"Splitting each input into {split_into} datapoints, resulting in {output_datapoints} samples")

X = np.zeros((output_datapoints, outputlength))
y = np.zeros((output_datapoints, 1))

for i, file in enumerate(inputfiles):
    sr, data = scipy.io.wavfile.read(inputpath + "/" + file)
    if sr != SR:
        raise(f"Samplerate of {file} is {sr} instead of {SR}")
    
    if len(data) < req_inputlength:
        raise(f"File {file} is not long enough")
    
    start_of_block = (len(data) - req_inputlength) // 2
    data_block = data[start_of_block:start_of_block+req_inputlength, 0]
    X[i*split_into:(i+1)*split_into, :] = data_block.reshape((split_into, outputlength))
    y[i*split_into:(i+1)*split_into, :] = int(file[3])

# normalize
x = X / np.max(np.abs(X))

if applyFFT:
    # TODO Hamming (or other) window?
    print("applying FFT")
    X = np.abs(np.fft.rfft(X))

print("Saving results")
np.save(outputpath + "/" + "split_samples_X.npy", X)
np.save(outputpath + "/" + "split_samples_y.npy", y)