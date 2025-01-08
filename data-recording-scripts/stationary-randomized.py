import numpy as np
import os
import audio_recorder
from ros_controller import ROSController


class robotRecording:
    def __init__(self, positions, starting_pose, jointangle_change, joint_change_mask, rec_length, data_dir_path="./data", rng_seed=None, notes=""):
        print("\n--- starting setup ---\n")
        self.positions = positions
        self.rec_length = rec_length
        self.starting_pose = starting_pose
        self.jointangle_change = jointangle_change
        self.joint_change_mask = joint_change_mask

        self.setup_data_dir(data_dir_path)
        print("- folder setup complete")
        self.notes = notes

        # TODO
        self.joint_limits_max = np.array([1,1,1,1,1,1,1])
        self.joint_limits_min = np.array([-1,-1,-1,-1,-1,-1,-1])

        self.rng = np.random.default_rng(rng_seed)

        self.ros_controller = ROSController()
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


    def record(self, completion_time=1):
        for i in range(self.positions):
            joint_pos = self.starting_pose
            print(f"\nmoving robot to {np.round(joint_pos, 2)}")
            self.ros_controller.move_to_position(joint_pos, completion_time)
            
            print("recording audio")
            self.record_sample(joint_pos)

            if self.positions > 1:
                print(f"Changing selected joint angles by  {self.jointangle_change} rad.")
                self.starting_pose += self.joint_change_mask * self.jointangle_change

            self.index += 1



    # currently not used
    def get_random_joint_position(self):
        return self.rng.random(7) * (self.joint_limits_max - self.joint_limits_min) + self.joint_limits_min
    
    def record_sample(self, joint_pos):
        sound_file = os.path.join(self.data_dir, f"{self.index}.wav")
        samples = self.audio_recorder.create_recording(self.rec_length, sound_file)

        joint_txt = ", ".join([str(q) for q in joint_pos])

        with open(self.sample_idx_file, "a") as f:
            f.write(f"{self.index}, {joint_txt}, {samples}, {self.audio_recorder.SR}, {self.session}, {self.notes}\n")

        with open(self.metadata_file, "w") as f:
                f.write(f"session: {self.session}\nindex: {self.index}")
        



if __name__ == "__main__":
    if positions := input("No. of sampled positions: "):
        positions = int(positions)
    else:
        positions = 1

    print("The default starting pose is [0, 0, 0, 0, 0, 0, 0]. Change this in the following if needed (e.g. 1,2,3,4,5,6,7).")
    if starting_pose := input("Starting pose: "):
        starting_pose = np.array([float(x) for x in starting_pose.split(",")])
    else:
        starting_pose = np.array([0, 0, 0, 0, 0, 0, 0])

    print("The default change of joint angle is 0.10 rad. Change this in the following if needed.")
    if jointangle_change := input("Change of joint angle: "):
        jointangle_change = float(jointangle_change)
    else:
        jointangle_change = 0.10

    print("Specify which joints to change (e.g. 1,1,1,0,0,0,0 for the first three joints). If not, all joints will be changed.")
    if joint_change_mask := input("Joint change mask: "):
        joint_change_mask = np.array([int(x) for x in joint_change_mask.split(",")])
    else:
        joint_change_mask = np.array([1, 1, 1, 1, 1, 1, 1])

    if rec_length := input("Recording lenghth in seconds: "):
        rec_length = int(rec_length)
    else:
        rec_length = 5

    if notes := input("Input notes: "):
        rec_length = int(rec_length)
    else:
        notes = ""

    record = robotRecording(positions, starting_pose, jointangle_change, joint_change_mask, rec_length, notes=notes)
    print("Stay clear of the workspace of the robot!")
    input("Press enter to start recording.")
    record.record()