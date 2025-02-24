# State Estimation with Acoustic Sensing
This git consists of two main programs. The folder data-recording contains 
the necessary scripts to control the robot and record audio data, while the folder
model-comparison contains the scripts to train and compare different machine learning models 
on that data.

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

TODO Respeaker anleitung?

### Architecture
The relevant modules are split into 4 files. ros_controller.py handles the communication
with the panda_ha instance on the control-PC. audio_recorder.py serves as a wrapper around
pyAudio and is responsible for recording and saving audiodata. The robot_movement_iterators.py
file offers different movement patterns the robot can execute to record data.
Finally, main.py coordinates those components, handles the CLI and takes care of the metadata.
