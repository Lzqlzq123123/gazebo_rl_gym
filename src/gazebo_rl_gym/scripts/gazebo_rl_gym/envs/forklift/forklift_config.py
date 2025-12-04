from gazebo_rl_gym.envs.base.robot_config import RobotCfg


class ForkliftCfg(RobotCfg):
    model = "forklift"
    controller_type = "skid_steer_drive_controller/SkidSteerDriveController"

    class action_space(RobotCfg.action_space):
        low = [-0.4, -2.0]
        high = [0.4, 2.0]