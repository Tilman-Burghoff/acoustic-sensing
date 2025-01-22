import numpy as np
import os
import audio_recorder
from ros_controller import ROSController
from robot_movement_iterators import Grid_2d
from time import sleep

DEBUG = False # if Robot is not available

class robotRecording:
    def __init__(self, position_iterator, rec_length, wait_for_move=False, data_dir_path="./data", rng_seed=None, notes="", debug=False):
        print("\n--- starting setup ---\n")
        self.rec_length = rec_length
        self.position_iterator = position_iterator
        self.wait_for_move = wait_for_move

        self.setup_data_dir(data_dir_path)
        print("- folder setup complete")
        self.notes = notes

        # TODO
        self.joint_limits_max = np.array([1,1,1,1,1,1,1])
        self.joint_limits_min = np.array([-1,-1,-1,-1,-1,-1,-1])

        self.rng = np.random.default_rng(rng_seed)

        self.ros_controller = ROSController(debug=debug)
        print("- ros setup complete")

        self.audio_recorder = audio_recorder.AudioRecorder()
        print("- audio setup complete")
        print("\n--- setup complete ---\n")
        print(f"starting session {self.session} at index {self.index}.")


    def setup_data_dir(self, data_dir_path):
        self.data_dir = os.path.normpath(data_dir_path)

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.sample_idx_file = os.path.join(self.data_dir, "samples.csv")
        self.metadata_file = os.path.join(self.data_dir, "metadata.txt")

        if not os.path.exists(self.sample_idx_file):
            with open(self.sample_idx_file, "x") as f:
                f.write("idx,q_0,q_1,q_2,q_3,q_4,q_5,q_6,samples,SR,session_id,notes\n")
            
            with open(self.metadata_file, "x") as f:
                f.write("session: 0\nindex:0")
            self.session = 0
            self.index = 0

        else:
            with open(self.metadata_file, "r") as f:
                metadata = f.readlines()
            
            self.session = int(metadata[0].split(":")[1]) + 1
            self.index = int(metadata[1].split(":")[1]) + 1
    
    def record(self):        
        for position, movetime, for_recording in self.position_iterator:
            print(f"moving robot to {np.round(position, 2)}")
            self.ros_controller.move_to_position(position, movetime)

            if self.wait_for_move or not for_recording:
                sleep(movetime)
            
            if for_recording:
                print("recording audio")
                self.record_sample(position)

            self.index += 1
            print()

    def record_sample(self, joint_pos):
        sound_file = os.path.join(self.data_dir, f"{self.index}.wav")
        samples = self.audio_recorder.create_recording(self.rec_length, sound_file)

        joint_txt = ", ".join([str(q) for q in joint_pos])

        with open(self.sample_idx_file, "a") as f:
            f.write(f"{self.index}, {joint_txt}, {samples}, {self.audio_recorder.SR}, {self.session}, {self.notes}\n")

        with open(self.metadata_file, "w") as f:
                f.write(f"session: {self.session}\nindex: {self.index}")
        


def select_catridge():
    # TODO select from different movement iterators
    selected = Grid_2d()
    print("The Parameters of this Iterator are:")
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
    print("The values are now:")
    for i, name in enumerate(selected.get_variable_names()):
        print(f"{i:2}: {name} = {selected.__getattribute__(name)}")
    selected.preview_iter()
    return selected.get_iterator()



if __name__ == "__main__":
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