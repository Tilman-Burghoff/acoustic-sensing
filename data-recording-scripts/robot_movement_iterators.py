import numpy as np
import matplotlib.pyplot as plt
import copy

class Grid_2d:
    def __init__(self):
        # TODO better names
        self.start_pos=np.array([-1.5708, 0, 0, -0.3, 0, 1.5708, 0])
        self.joint_x=0
        self.joint_y=3
        self.points_x=32
        self.points_y=26
        self.step_x=0.1
        self.step_y=-0.1
        self.continue_from=0
        self.public_variable_names = ['start_pos', 'joint_x', 'joint_y', 
                                      'points_x', 'points_y', 'step_x', 
                                      'step_y', 'continue_from']
        
    def get_variable_names(self):
        return self.public_variable_names
    
    def preview_iter(self):
        xs = []
        ys = []
        for pose, _, record in self.get_iterator(log_index=False):
            if record:
                xs.append(pose[self.joint_x])
                ys.append(pose[self.joint_y])

        plt.plot(xs, ys)
        plt.xlabel("q0")
        plt.ylabel("q3")
        plt.axis("equal")
        plt.title("Motion Preview")
        plt.show()

    def get_position_by_index(self, idx):
        y_mult = idx // self.points_x
        x_mult = idx % self.points_x if y_mult % 2 == 0 else self.points_x -1 - (idx % self.points_x)
        self.pose[self.joint_x] = self.start_pos[self.joint_x] + x_mult * self.step_x
        self.pose[self.joint_y] = self.start_pos[self.joint_y] + y_mult * self.step_y
        return self.pose
    
    # TODO: moving time = 5s for starting position 
    def get_iterator(self, log_index=True):
        self.pose = copy.copy(self.start_pos)
        def grid_2d_iter():
            positions = self.points_x * self.points_y
            if self.continue_from == 0:
                yield self.start_pos, 5, False
            else:
                pose = self.get_position_by_index(self.continue_from - 1)
                yield pose, 5, False
            for i in range(self.continue_from, positions):
                if i == 0:
                    print(f"Moving to position_index {i}")
                    pose = self.get_position_by_index(i)
                    yield pose, 5, True
                if log_index:
                    print(f"Moving to position_index {i}")
                pose = self.get_position_by_index(i)
                yield pose, 1, True

        return grid_2d_iter()