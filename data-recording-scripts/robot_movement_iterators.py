import numpy as np
import matplotlib.pyplot as plt
import copy
from abc import ABC, abstractmethod

class MoveIter(ABC):
    public_variable_parsers = {}

    def get_variable_names(self):
        return list(self.public_variable_parsers.keys())
    
    def set_public_var(self, name, value):
        assert name in self.public_variable_parsers
        assert type(value) == str
        value = self.public_variable_parsers[name](value)
        return self.__setattr__(name, value)
    
    @abstractmethod
    def preview_iter(self):
        pass

    @abstractmethod
    def get_iterator(self):
        pass
    
    #parsers
    def parse_jointpos(self, value: str):
        assert value.startswith("np.array([")
        assert value.endswith("])")
        vals = [float(val) for val in value[10:-2].split(",")]
        assert len(vals) == 7
        return np.array(vals)
    
    def parse_jointnum(self, value):
        joint = int(value)
        assert 0 <= joint <= 6
        return joint
    
    def pos_int(self, value):
        val = int(value)
        assert 0 < val
        return val
    
    def nonneg_int(self, value):
        val = int(value)
        assert 0 <= val
        return val


class Move_Once(MoveIter):
    def __init__(self):
        self.move_to = np.array([0, 0, 0, -1.5708, 0, 1.5708, 0])
        self.move_time_s = 5
        self.public_variable_parsers = {
            "move_to": self.parse_jointpos,
            "move_time_s": self.pos_int}

    def preview_iter(self):
        pass

    def get_iterator(self):
        def moveiter():
            yield self.move_to, self.move_time_s, False
        return moveiter()


class Grid_2d(MoveIter):
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
        self.public_variable_parsers = {
            'start_pos': self.parse_jointpos,
            'joint_x': self.parse_jointnum,
            'joint_y': self.parse_jointnum, 
            'points_x': self.pos_int,
            'points_y': self.pos_int, 
            'step_x': float, 
            'step_y': float, 
            'continue_from': self.nonneg_int
        }
        
    
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
                # if i == self.continue_from:
                #    print(f"Moving to position_index {i}")
                #    pose = self.get_position_by_index(i)
                #    yield pose, 5, False
                if log_index:
                    print(f"Moving to position_index {i}")
                pose = self.get_position_by_index(i)
                yield pose, 1, True

        return grid_2d_iter()