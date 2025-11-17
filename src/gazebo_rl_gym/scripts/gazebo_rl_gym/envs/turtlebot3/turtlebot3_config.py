from gazebo_rl_gym.envs.base.robot_config import RobotCfg


class Turtlebot3Cfg(RobotCfg):
    model = "turtlebot3_waffle"
    controller_type = "velocity_controllers/JointVelocityController"

    class action_space(RobotCfg.action_space):
        low = [-0.26, -1.82]
        high = [0.26, 1.82]
