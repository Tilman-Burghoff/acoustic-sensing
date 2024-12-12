import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
from hybrid_automaton_msgs import srv


class ROSController:
    def __init__(self):
        
        rospy.init_node('ros_controller', anonymous=False)
        
        self.joint_goal_pub = rospy.Publisher('/joint_goal', Float64MultiArray, queue_size=10)
    
        self.call_ha = rospy.ServiceProxy('update_hybrid_automaton', srv.UpdateHybridAutomaton)
        
        self.kp = [60.0, 60.0, 60.0, 60.0, 60.0, 20.0, 20.0] 
        self.kv = [30.0, 30.0, 20.0, 20.0, 20.0, 20.0, 10.0]
        
        self.current_joint_states = None
        rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)


    def joint_states_callback(self, msg: JointState):
 
        self.current_joint_states = {
            "name": msg.name,
            "position": msg.position,
            "velocity": msg.velocity,
            "effort": msg.effort
        }
        rospy.loginfo(f"Received joint states: {self.current_joint_states}")


    def get_current_joint_positions(self):
 
        if self.current_joint_states:
            return self.current_joint_states["position"]
        else:
            rospy.logwarn("Joint states not received yet!")
            return None


    def list_to_ha_string(self, array: 'list[float]') -> str:

        """Return a string that correspond to a list of float for the Hybrid Automaton"""
        ha_string = '[{NUM_ROWS}, 1] {ARRAY}'.format(NUM_ROWS=len(array), ARRAY='; '.join(str(x) for x in array))
        return ha_string


    def create_HA(self, joint_pos: 'list[float]', completion_times: 'list[float]') -> str:
        
        """Return a string in xml format for the Hybrid Automaton"""
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


    def move_to_position(self, joint_pos, completion_time=4):
  
        ha_to_send = self.create_HA(joint_pos=joint_pos, completion_times=[completion_time])
        self.call_ha(ha_to_send)
        rospy.loginfo(f"Moved robot to position: {joint_pos}")


    def publish_joint_goal(self, joint_pos):
        
        # sends topic joint_goal
        joint_goal = Float64MultiArray()
        joint_goal.layout.dim.append(MultiArrayDimension())
        joint_goal.layout.dim[0].label = "joint_q"
        joint_goal.layout.dim[0].size = len(joint_pos)

        joint_goal.data = joint_pos

        self.joint_goal_pub.publish(joint_goal)
        rospy.loginfo(f"Published joint goal: {joint_pos}")
