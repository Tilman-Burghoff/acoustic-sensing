# This file provides an interface for controlling the Panda arm using ROS and 
# the Panda Hybrid Automaton Controller from the RBO Lab.
# The ROSController class initializes ROS nodes, handles state collection,
# and provides methods for publishing joint goals and sending commands
# to the Panda Hybrid Automaton. This allows the robot to execute movements
# based on predefined joint positions.

import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from hybrid_automaton_msgs import srv
from panda_ha_msgs.msg import RobotState


class ROSController:
    """
    This class provides an interface for controlling a robot using ROS.
    It initializes ROS communication, collects joint states, and sends
    commands to move the robot using Panda Hybrid Automaton.
    
    If debug mode is enabled, ROS nodes and publishers are not initialized.
    """
    def __init__(self, debug=False):
        self.debug = debug

        if not self.debug:
            self.init_rospy()

        self.collecting = True

    def init_rospy(self):
        """Initializes the ROS node and sets up publishers and service clients."""
        rospy.init_node('ros_controller', anonymous=False)
        
        # Publisher for sending joint goals
        self.joint_goal_pub = rospy.Publisher('/joint_goal', Float64MultiArray, queue_size=10)
    
        # Service client for updating the Hybrid Automaton
        self.call_ha = rospy.ServiceProxy('update_hybrid_automaton', srv.UpdateHybridAutomaton)
        
        # Gains for the controller
        self.kp = [60.0, 60.0, 60.0, 60.0, 60.0, 20.0, 20.0] 
        self.kv = [30.0, 30.0, 20.0, 20.0, 20.0, 20.0, 10.0]
        
        # Joint positions will be stored here
        self.q = None

        # Subscriber for receiving current state
        rospy.Subscriber('/robot/currentstate', RobotState, self.collectStateCallback)


    def collectStateCallback(self, data):
        """Callback function to store the latest joint positions."""
        if not self.collecting:
            return
        self.q = data.q

    def get_current_joint_positions(self):
        """Returns the current joint positions if available, otherwise logs a warning."""
        if self.q:
            return self.q
        else:
            rospy.logwarn("Joint states not received yet!")
            return None


    def list_to_ha_string(self, array: 'list[float]') -> str:
        """Converts a list of floats into a string formatted for the Hybrid Automaton."""
        ha_string = '[{NUM_ROWS}, 1] {ARRAY}'.format(NUM_ROWS=len(array), ARRAY='; '.join(str(x) for x in array))
        return ha_string


    def create_HA(self, joint_pos: 'list[float]', completion_times: 'list[float]') -> str: 
        """Returns a string in xml format for the Hybrid Automaton"""
        xml = '<HybridAutomaton current_control_mode="Mode0" name="Sequence">'\
              '    <ControlMode name="Mode0">'\
              '        <ControlSet type="pandaControlSet" name="default">'\
              '            <Controller type="JointPositionController" name="MainController"'\
              '                        goal="{JOINT_POSITION}"'\
              '                        kp="{KP}"'\
              '                        kv="{KV}"'\
              '                        completion_times="{COMPLETION_TIMES}"'\
              '                        a_max="[0,0]" priority="0"'\
              '                        interpolation_type="cubic"'\
              '            />'\
              '        </ControlSet>'\
              '    </ControlMode>'\
              '</HybridAutomaton>'
        
        xml = xml.format(
            JOINT_POSITION=self.list_to_ha_string(joint_pos),
            KP=self.list_to_ha_string(self.kp),
            KV=self.list_to_ha_string(self.kv),
            COMPLETION_TIMES=self.list_to_ha_string(completion_times),
        )
        return xml


    def move_to_position(self, joint_pos, completion_time=1):
        """Sends a command to move the robot to a specified joint position."""
        if not self.debug:
            ha_to_send = self.create_HA(joint_pos=joint_pos, completion_times=[completion_time])
            self.call_ha(ha_to_send)
            rospy.loginfo(f"Moved robot to position: {joint_pos}")


    def publish_joint_goal(self, joint_pos):        
        """Publishes a joint goal message to the appropriate ROS topic."""
        joint_goal = Float64MultiArray()
        joint_goal.layout.dim.append(MultiArrayDimension())
        joint_goal.layout.dim[0].label = "joint_q"
        joint_goal.layout.dim[0].size = len(joint_pos)

        joint_goal.data = joint_pos

        self.joint_goal_pub.publish(joint_goal)
        rospy.loginfo(f"Published joint goal: {joint_pos}")