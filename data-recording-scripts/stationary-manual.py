import numpy as np
import pandas as pd
import scipy.io.wavfile
import os
from jacktools.jacksignal import JackSignal
from ros_controller import ROSController


class robotRecording:
    def __init__(self, positions, rec_length_smpls, data_dir_path="./data", rng_seed=None, notes=""):
        print("\n--- starting setup ---\n")
        self.positions = positions
        self.rec_length_smpls = rec_length_smpls

        self.setup_data_dir(data_dir_path)
        print("- folder setup complete")
        self.notes = notes

        self.joint_limits_max = np.array([1,1,1,1,1,1,1])
        self.joint_limits_min = np.array([-1,-1,-1,-1,-1,-1,-1])

        self.rng = np.random.default_rng(rng_seed)

        self.ros_controller = ROSController()
        print("- ros setup complete")

        print("- ros setup complete")
        self.setup_jack()
        print("- jacktools setup complete")
        print("\n--- setup complete ---\n")
        print(f"starting session {self.session} at index {self.index}.")


    def setup_jack(self):
        self.J = JackSignal("JS")
        assert self.J.get_state() >= 0, "Creating JackSignal failed."
        name, sr, period = self.J.get_jack_info()
        self.SR = sr
        
        self.J.create_input(1, "in_1")
        self.J.connect_input(1, "system:capture_1")
        self.J.silence()

        self.Ain = np.zeros(self.rec_length_smpls, dtype=np.float32)
        self.J.set_input_data(1, self.Ain)


    def setup_data_dir(self, data_dir_path):
        self.data_dir = os.path.normpath(data_dir_path)

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.sample_idx_file = os.path.join(self.data_dir, "samples.csv")
        self.metadata_file = os.path.join(self.data_dir, "metadata.txt")

        if not os.path.exists(self.sample_idx_file):
            with open(self.sample_idx_file, "x") as f:
                f.write("idx, q_0, q_1, q_2, q_3, q_4, q_5, q_6, samples, SR, session_id, notes\n")
            
            with open(self.metadata_file, "x") as f:
                f.write("session: 0\nindex:0")
            self.session = 0
            self.index = 0

        else:
            with open(self.metadata_file, "r") as f:
                metadata = f.readlines()
            
            self.session = int(metadata[0].split(":")[1]) + 1
            self.index = int(metadata[1].split(":")[1]) + 1


    def get_joint_position(self):
        current_positions = self.ros_controller.get_current_joint_positions()
        if current_positions:
            print(f"Current joint positions: {current_positions}")
        return current_positions

    def record(self):
        for i in range(self.positions):
            input("Move robot into position, then press enter.")
            joint_pos = self.get_joint_position()
            print("recording audio")
            self.record_sample(joint_pos)
            self.index += 1

    def record_sample(self, joint_pos):
        self.J.process()
        self.J.wait()
        sound_file = os.path.join(self.data_dir, f"{self.index}.wav")
        scipy.io.wavfile.write(sound_file, self.SR, self.Ain)

        joint_txt = ", ".join([str(q) for q in joint_pos])
        
        with open(self.sample_idx_file, "a") as f:
            f.write(f"{self.index}, {joint_txt}, {self.rec_length_smpls}, {self.SR}, {self.session}, {self.notes}\n")

        with open(self.metadata_file, "w") as f:
                f.write(f"session: {self.session}\nindex: {self.index}")
        



if __name__ == "__main__":
    if positions := input("No. of sampled positions: "):
        positions = int(positions)
    else:
        positions = 1
    if rec_length := input("Recording lenghth in samples: "):
        rec_length = int(rec_length)
    else:
        rec_length = 48000 * 5

    if notes := input("Input notes: "):
        rec_length = int(rec_length)
    else:
        notes = ""

    record = robotRecording(positions, rec_length, notes=notes)
    print("Stay clear of the workspace of the robot!")
    input("Press enter to start recording.")
    record.record()