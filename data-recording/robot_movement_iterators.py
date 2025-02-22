# This file provides multiple different movement patterns for the robot
# Each Movement pattern is implemented as a constructor class, which returns
# an Iterator when calling get_iterator. The iterator returns a
# tuple (np.array, number, bool) in each iteration, which contains the new
# pose (as 7 DOF joint angles in rad), how long the robot should move there
# and whether audio should be recorded at that pose. Those Iterators should
# be interchangeable, therefore thair common api is specified in the abstract
# base class MoveIter.


import numpy as np
import copy
from abc import ABC, abstractmethod
from typing import List, Iterator, Tuple, Callable, Any, NewType
from numbers import Real as RealNumber

JointPos = NewType("JointPos", Any) # Type For the 7 DOF joint pos

class MoveIter(ABC):
    """
    This class functions as a template for the api each constructor
    should implement. Note that due to some revisions the only method
    each constructor needs to implement is get_iterator.

    Each constructor exposes its hyperparameters via 
    get_variable_names, which can then be changed by 
    set_public_var, which accepts a valid python expression 
    for that variable's type.

    Once the user is happy with the parameters, an iterator can
    be obtained with get_iterator. This function should always 
    return the same iterator, as long as the hyperparameters aren't
    changed between two function calls. Changing hyperparameters between
    Iterators is undefined behavior, and a new constructor object should
    be used instead.

    Other then functioning as a template, this class also implements
    the get and set methods as well as some useful parsers.
    """

    public_variable_parsers = {} # Variable names and their parsers.

    def get_variable_names(self) -> List[str]:
        """ returns a list of the public hyperparameters"""
        return list(self.public_variable_parsers.keys())
    
    def set_public_var(self, name: str, value: str):
        """Sets a hyperparameter, as long as the parser accepts it"""
        assert name in self.public_variable_parsers
        assert type(value) == str
        value = self.public_variable_parsers[name](value)
        return self.__setattr__(name, value)

    @abstractmethod
    def get_iterator(self) -> Iterator[Tuple[JointPos, RealNumber, bool]]:
        """To be implemented by each constructor.
        
        Returns an Iterator iterating over the robots positions
        with a move_time number >0 and a recording bool"""
        ...
    
    # parsers
    def parse_jointpos(self, value: str) -> JointPos:
        """Parse string into 7-DOF joint pos"""
        assert value.startswith("np.array([")
        assert value.endswith("])")
        vals = [float(val) for val in value[10:-2].split(",")]
        assert len(vals) == 7
        return np.array(vals)
    
    def parse_jointnum(self, value: str) -> int:
        """Parse string into int i with 0 <= i <= 6"""
        joint = int(value)
        assert 0 <= joint <= 6
        return joint
    
    def factory_int_greater(self, lowerbound: int) -> Callable[[str], int]:
        """Returns int-parser which checks that int > lowerbound"""
        def int_greater(value: str) -> int:
            val = int(value)
            assert lowerbound < val
            return val
        return int_greater
    
    def pos_int(self, value: str) -> int:
        val = int(value)
        assert 0 < val
        return val
    
    def nonneg_int(self, value: str) -> int:
        val = int(value)
        assert 0 <= val
        return val
    


class Move_Once(MoveIter):
    """Moves robot to move_to in move_time seconds without recording audio"""
    def __init__(self):
        self.move_to = np.array([0, 0, 0, -1.5708, 0, 1.5708, 0])
        self.move_time_s = 5
        self.public_variable_parsers = {
            "move_to": self.parse_jointpos,
            "move_time_s": self.pos_int}

    def get_iterator(self):
        def moveiter():
            yield self.move_to, self.move_time_s, False
        return moveiter()


class Line(MoveIter):
    """Moves robot from start_point to end_point while 
    recording at sample_points many linearly spaced spots
    along the way. continue_from can be used to start from a 
    position along the way by specifying an index.
    """
    def __init__(self):
        self.start_point = np.array([0, 0, 0, -0.3, 0, 1.5708, 0])
        self.end_point = np.array([0, 0, 0, -2.8, 0, 1.5708, 0])
        self.sample_points = 50
        self.continue_from = 0
        self.public_variable_parsers = {
            "start_point": self.parse_jointpos,
            "end_point": self.parse_jointpos,
            "sample_points": self.factory_int_greater(1),
            "continue_from": self.nonneg_int
        }

    def get_pose_by_idx(self, idx):
        return self.start_point + (self.end_point - self.start_point) * idx/(self.sample_points - 1)

    def get_iterator(self, log_index=True):
        def lineiter():
            startpose = self.get_pose_by_idx(self.continue_from)
            yield startpose, 5, False
            for i in range(self.continue_from, self.sample_points):
                pose = self.get_pose_by_idx(i)
                if log_index:
                    print(f"Moving to position_index {i}")
                yield pose, 1, True
        return lineiter()


class Grid_2d(MoveIter):
    """Moves robot through a 2d grid in snake like fashion, while keeping
    all joints except for joint_x and joint_y in the place specified in start_pos.
    Moves the two joints by step_x or step_y radians for points_x or points_y steps
    starting fro start_pos. continue_from can be used to start from a 
    position along the way by specifying an index.
    """
    def __init__(self):
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

    def get_position_by_index(self, idx):
        y_mult = idx // self.points_x
        x_mult = idx % self.points_x if y_mult % 2 == 0 else self.points_x -1 - (idx % self.points_x)
        self.pose[self.joint_x] = self.start_pos[self.joint_x] + x_mult * self.step_x
        self.pose[self.joint_y] = self.start_pos[self.joint_y] + y_mult * self.step_y
        return self.pose
     
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
                if log_index:
                    print(f"Moving to position_index {i}")
                pose = self.get_position_by_index(i)
                yield pose, 1, True

        return grid_2d_iter()
    

class Random_Uniform(MoveIter):
    """Moves the robot to no_samples many randomly generated positions, which
    lie inbetween min_joint_bound and max_joint_bound. A seed can be specified if needed.
    continue_from can be used to start from a position along the way by specifying an index.
    """
    def __init__(self):
        self.min_joint_bound = np.array([-1.5708, 0, 0, -0.3, 0, 1.5708, 0])
        self.max_joint_bound = np.array([1.5708, 0, 0, -2.6, 0, 1.5708, 0])
        self.no_samples = 50
        self.seed = None
        self.continue_from = 0
        self.generated_seed = None
        self.public_variable_parsers = {
            "min_joint_bound": self.parse_jointpos,
            "max_joint_bound": self.parse_jointpos,
            "no_samples": self.nonneg_int,
            "seed": int,
            "continue_from": self.nonneg_int
        }

    def generate_seed(self):
        if self.seed is not None:
            return self.seed
        if self.generated_seed is not None:
            return self.generated_seed
        self.generated_seed = np.random.default_rng().integers(0, 1000000) # 6 stelliger seed
        return self.generated_seed
    
    def get_iterator(self, log_index=True):
        seed = self.generate_seed()
        rng = np.random.default_rng(seed)

        if self.continue_from > 0:
            for _ in range(self.continue_from):
                _ = rng.uniform(self.min_joint_bound, self.max_joint_bound)
        
        if log_index:
            print(f"Drawing Samples with seed {seed}")
        
        def random_iter():
            for i in range(self.continue_from, self.no_samples):
                rand = rng.uniform(self.min_joint_bound, self.max_joint_bound)
                if log_index:
                    print(f"Generating sample with sample_index {i}")
                yield rand, 5, False
        return random_iter()