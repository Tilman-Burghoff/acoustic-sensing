import numpy as np
import os
from audio_recorder import AudioRecorder# Wrapper around pyaudio
from ros_controller import ROSController # Wrapper around RosPy
from robot_movement_iterators import Grid_2d, Move_Once, Line, Random_Uniform
from time import sleep

DEBUG = False # set to True for testing if robot is not available

class robotRecording:
    def __init__(self, 
            position_iterator,       # Iterator returning (position, time, record_bool) tuple
            rec_length,              # Recording Length in seconds
            wait_for_move=False,     # Whether to record audio as soon as movement command is issued
            data_dir_path="./data",  # Directory for recordings
            notes="",                # notes to add to samples.csv
            debug=False              # set to True if robot not available
        ):
        print("\n--- starting setup ---\n")
        self.rec_length = rec_length
        self.position_iterator = position_iterator
        self.wait_for_move = wait_for_move

        self.setup_data_dir(data_dir_path)
        print("- folder setup complete")
        self.notes = notes.replace(",", " - ") # to be compatible with csv

        self.ros_controller = ROSController(debug=debug)
        print("- ros setup complete")

        self.audio_recorder = AudioRecorder()
        print("- audio setup complete")
        print("\n--- setup complete ---\n")
        print(f"starting session {self.session} at index {self.index}.")


    def setup_data_dir(self, data_dir_path):
        """Creates Data Dir and metadata-structure if they don't exist
        loads metadata (recording session and recording index) otherwise.
        """
        self.data_dir = os.path.normpath(data_dir_path)

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.sample_idx_file = os.path.join(self.data_dir, "samples.csv")
        self.metadata_file = os.path.join(self.data_dir, "metadata.txt")

        if not os.path.exists(self.sample_idx_file):
            # create sample metadata (samples.csv) 
            with open(self.sample_idx_file, "x") as f:
                f.write("idx,q_0,q_1,q_2,q_3,q_4,q_5,q_6,samples,SR,session_id,notes\n")
        
        if not os.path.exists(self.metadata_file):
            # create recording metadata (metadata.txt)
            with open(self.metadata_file, "x") as f:
                f.write("session: 0\nindex:0")
            self.session = 0
            self.index = 0

        else:
            # read recording metadata
            with open(self.metadata_file, "r") as f:
                metadata = f.readlines()
            
            self.session = int(metadata[0].split(":")[1]) + 1
            self.index = int(metadata[1].split(":")[1]) + 1
    
    def record(self):
        """Use position_iterator to move robots to requiered positions
        and record audio (if needed)
        """     
        for position, movetime, record_bool in self.position_iterator:
            print(f"moving robot to {np.round(position, 2)}")
            self.ros_controller.move_to_position(position, movetime)

            # move_to_position doesn't wait until move is done (we only send a message to controller)
            # therefore we have to manually wait the requiered time (if needed)
            if self.wait_for_move or not record_bool or movetime < self.rec_length:
                # we get into trouble if we send a movement command 
                # before the previous one is executed
                sleep(movetime)
            
            if record_bool:
                print("recording audio")
                self.record_sample(position)

            self.index += 1
            print()

    def record_sample(self, joint_pos):
        """Records a sample and writes the nessecary metadata"""
        sound_file = os.path.join(self.data_dir, f"{self.index}.wav")
        samples = self.audio_recorder.create_recording(self.rec_length, sound_file)

        joint_txt = ", ".join([str(q) for q in joint_pos])

        with open(self.sample_idx_file, "a") as f:
            f.write(f"{self.index}, {joint_txt}, {samples}, {self.audio_recorder.SR}, {self.session}, {self.notes}\n")

        with open(self.metadata_file, "w") as f:
                f.write(f"session: {self.session}\nindex: {self.index}")
        


def select_catridge():
    """Makes user select a movement pattern 'cartridge' with CLI."""

    # Select cartridge
    cartridges = {
        "Move Once": Move_Once(),
        "Move and Sample Line": Line(),
        "Sample 2d Grid": Grid_2d(),
        "Sample Uniform Random Positions": Random_Uniform()
    }
    cart_list = list(cartridges.keys())
    print("Available Movement Patterns:")
    for i, cart in enumerate(cart_list):
        print(f"{i:2}: {cart}")
    cart_idx = int(input("Select Movement-pattern by index: "))
    selected = cartridges[cart_list[cart_idx]]

    print(f"Selected Pattern: {cart_list[cart_idx]}\n{selected.__doc__}\n")

    # Change Parameters
    print("The Parameters of this Movement Pattern are:")
    parameterlist = selected.get_variable_names()
    for i, name in enumerate(parameterlist):
        print(f"{i:2}: {name} = {selected.__getattribute__(name)}")
    
    change_vars = True
    while change_vars:
        if var_idx := input("Index of variable to change (enter to finish): "):
            name = parameterlist[int(var_idx)]
            val = input(f"{name} = ")
            selected.set_public_var(name, val)
        else:
            change_vars = False
    
    print("Selected Values:")
    for i, name in enumerate(selected.get_variable_names()):
        print(f"{i:2}: {name} = {selected.__getattribute__(name)}")
    return selected.get_iterator()



if __name__ == "__main__":
    # CLI to make the user select the necessary parameters
    iterator = select_catridge()

    if rec_length := input("Recording length in seconds: "):
        rec_length = int(rec_length)
    else:
        rec_length = 5

    if notes := input("Input notes: "):
        rec_length = int(rec_length)
    else:
        notes = ""

    record = robotRecording(iterator, rec_length, notes=notes, debug=DEBUG)
    print("Stay clear of the workspace of the robot!")
    input("Press enter to start recording.")
    record.record()