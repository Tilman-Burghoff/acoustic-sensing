# State Estimation with Acoustic Sensing
This git consists of two main programs. The folder data-recording contains 
the necessary scripts to control the robot and record audio data, while the folder
model-comparison contains the scripts to train and compare different machine learning models 
on that data. Finally, the results folder contains the outputs of the model-comparison script 
on the four datasets we collected (different microphone positions) as well as some notebooks 
showing examples of how to compare them.

## Data Recording
The main.py function implements a CLI that should be straight-forward to use.
If the panda robot is not available, set the DEBUG flag in the beginning of main.py
to true. 

### Setup
1. Connect your computer and the robot control PC to the same network.
2. Find your and the robots IP.
3. Set ROS_IP on both machines 
4. Set ROS_MASTER_URI=\[controllerIP\]:11311 on your machine
5. Start hybrid automation and check whether the connection works

Now the robot controller should work. Make sure you select the right audio input in the CLI.

If you want to use the ReSpeaker, make sure that the firmware is up to data and uses the 6-channel version.
A guide on how to update it can be found [here](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/#update-firmware).

The mic output on channel 0 of the ReSpeaker is preprocessed with among other things a noise suppression algorithm.
This is detrimental in our case. The two ways around it are to either use the raw mic input on channel 1-4, 
like we did, or to disable as many of the internal dsp processing as possible.
This can be done by running the config_usb4_mic_array.sh script in the usb_4_mic_array folder created during
updating the firmware. 

Make sure that you source your ros installation and export the relevant IPs in the terminal where you run the 
main.py script.

### Architecture
The relevant modules are split into 4 files. ros_controller.py handles the communication
with the panda_ha instance on the control-PC. audio_recorder.py serves as a wrapper around
pyAudio and is responsible for recording and saving audio. The robot_movement_iterators.py
file offers different movement patterns the robot can execute to record data.
Finally, main.py coordinates those components, handles the CLI and takes care of the metadata.


## Model Comparison
This program compares 4 different model architectures: k nearest neighbor regression, linear regression,
a dense neural network and a convolutional neural network. During an earlier iteration with easier data 
(only one joint was moved) we also used a support vector machine, but there sklearns implementation 
wasn't able to find a solution in 10+ minutes.

The program performs a 10-fold train-test split, and writes the raw output of the models on the test data 
into a csv-file.

### Setup
Copy the recorded data into the model-comparison folder and run main.py in that context. If you do not use the 
standard names, you have to give the foldername and the path to the sample metadata csv-file as arguments to 
the read_data() function in main.py.

### Architecture
Models are implemented in model_testing_interface.py. There, all models are implemented as a class with a train
and a test function as members. Possible hyperparameter can be given during initialization, but the choice is
kept to a minimum. All models inherit the Model abstract base class.

The data_utils.py script provides functions to load data and perform a train-test split, as well as to save the
results to a csv file.