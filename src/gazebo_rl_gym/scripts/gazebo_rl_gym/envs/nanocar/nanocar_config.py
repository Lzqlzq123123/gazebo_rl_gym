from gazebo_rl_gym.envs.base.robot_config import RobotCfg


class NanocarCfg(RobotCfg):
    model = "nanocar"
    controller_type = "velocity_controllers/JointVelocityController"

    class action_space(RobotCfg.action_space):
        low = [-0.4, -2.0]
        high = [0.4, 2.0]
