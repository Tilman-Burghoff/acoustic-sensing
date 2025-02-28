# Robot State Estimation with Acoustic Sensing
This project consists of two main programs. The folder data-recording contains 
the necessary scripts to control the robot and record audio data, while the folder
model-comparison contains the scripts to train and compare different machine learning models 
on that data. 
Finally, the results folder contains the outputs of the model-comparison script 
on the four datasets we collected (different microphone positions) as well as some notebooks 
showing examples of how to compare them.
Additional experiments we conducted can be found in the other-experiments folder. Those are not necessary
to understand our main results. While they might provide additional insight, we can't guarantee that all
of them work with the listed requirements (for example because they contain syntax from python >= 3.10).

## Prerequisites
This project requires a Python version less than 3.10.
```sh
python --version  # Ensure it's below 3.10
```
### Requirements  
1. Install Panda Hybrid Automaton Manager.
2. Install packages hybrid_automaton_msgs and panda_ha_msgs.
3. Install dependencies.
```sh
pip install -r requirements.txt
```

## Data Recording
The main.py function implements a CLI that should be straight-forward to use.
If the panda robot is not available, set the DEBUG flag in the beginning of main.py
to true. 

### Setup
1. Connect your computer and the robot control PC to the same network.
2. Find your and the robots IP.
```sh
ifconfig
```
3. Set ROS_IP on both machines .
4. Set ROS_MASTER_URI=\[controllerIP\]:11311 on your machine.
5. Launch Panda Hybrid Automaton and check whether the connection works.
```sh
roslaunch panda_hybrid_automaton_manager panda_ha.launch
```

Now the robot controller should work. Make sure you select the right audio input in the CLI.

If you want to use the ReSpeaker, make sure that the firmware is up to data and uses the 6-channel version.
A guide on how to update it can be found [here](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/#update-firmware).

The mic output on channel 0 of the ReSpeaker is preprocessed with among other things a noise suppression algorithm.
This is detrimental in our case. The two ways around it are to either use the raw mic input on channel 1-4, 
like we did, or to disable as many of the internal dsp processing as possible.
This can be done by running the config_usb4_mic_array.sh script in the usb_4_mic_array folder created during
updating the firmware. 

Make sure that you source your ros installation and export the relevant IPs in the terminal where you run 
```sh
python3 main.py
```

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

Note that while data-recording saves all joint positions, these programs are hard-coded to use the jointangles
$$q_0$$ and $$q_3$$ as inputs, since we focused on them. This however can be easily changed in the relevant 
functions in data_utils.py (read_data change the jont angles used and create_outputfile to change the csv-header).

### Setup
Copy the recorded data into the model-comparison folder and run
```sh
python3 main.py
```

If you do not use the standard names, you have to give the foldername and the path to the sample 
metadata csv-file as arguments to the read_data() function in main.py.

### Architecture
Models are implemented in model_testing_interface.py. There, all models are implemented as a class with a train
and a test function as members. Possible hyperparameter can be given during initialization, but the choice is
kept to a minimum. All models inherit the Model abstract base class.

The data_utils.py script provides functions to load data and perform a train-test split, as well as to save the
results to a csv file.


## Detailed Analysis
For further analysis on our data or earlier experiments, have a look at the detailed-analysis folder,
However, you do not need these scripts for the program to work. Our most important results are in the results folder.

## Datasets
You can find our collected data in this [folder](https://tubcloud.tu-berlin.de/s/3YSTCpWXXaTaM9S).
