#!/usr/bin/env python3
import os
import subprocess

import rospy
import yaml
import xacro
import rosparam
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnModel

from gazebo_rl_gym.utils.config_loader import load_environment_config
from gazebo_rl_gym.utils.path_utils import resolve_relative
from tf.transformations import quaternion_from_euler

def main():
    rospy.init_node('robot_spawner')
    
    config_file_path = rospy.get_param('~config_file')
    controllers_file_path = rospy.get_param('~controllers_file', resolve_relative('config', 'controllers.yaml'))

    env_config = load_environment_config(config_file_path)

    controllers_config = {}
    if os.path.exists(controllers_file_path):
        with open(controllers_file_path, 'r', encoding='utf-8') as handle:
            controllers_config = yaml.safe_load(handle)

    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    spawn_model_prox = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)

    for scenario in env_config.robots:
        spec = scenario.spec
        cfg = spec.cfg

        robot_name = scenario.name
        model_name = cfg.model
        if not model_name:
            raise ValueError(f"Robot preset '{scenario.preset}' must define a URDF/Xacro model name")
        pose_dict = scenario.pose

        pose = Pose()
        pose.position.x = pose_dict.get('x', 0.0)
        pose.position.y = pose_dict.get('y', 0.0)
        pose.position.z = pose_dict.get('z', 0.0)

        q = quaternion_from_euler(0, 0, pose_dict.get('yaw', 0.0))
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        urdf_path = _resolve_model_path(model_name)

        if urdf_path.endswith('.xacro'):
            robot_description = xacro.process_file(urdf_path).toxml()
        else:
            with open(urdf_path, "r", encoding="utf-8") as handle:
                robot_description = handle.read()

        rospy.loginfo(f"Spawning {robot_name} ({model_name})...")
        # 完成单个机器人在仿真中的注册与摆放
        spawn_model_prox(
            model_name=robot_name,
            model_xml=robot_description,
            robot_namespace=robot_name,
            initial_pose=pose,
            reference_frame="world"
        )
        rospy.loginfo(f"{robot_name} spawned.")

        controller_config = controllers_config.get(robot_name)
        if controller_config is None and scenario.preset in controllers_config:
            controller_config = controllers_config[scenario.preset]
        _spawn_controllers(robot_name, controller_config or {})


def _resolve_model_path(model_name: str) -> str:
    candidates = [
        resolve_relative('urdf', f"{model_name}.xacro"),
        resolve_relative('urdf', f"{model_name}.urdf"),
        resolve_relative('urdf', f"{model_name}.urdf.xacro"),
        resolve_relative('urdf', model_name, f"{model_name}.xacro"),
        resolve_relative('urdf', model_name, f"{model_name}.urdf"),
        resolve_relative('urdf', model_name, f"{model_name}.urdf.xacro"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Unable to locate URDF/Xacro for model '{model_name}'")


def _spawn_controllers(robot_name: str, controllers: dict) -> None:
    if not controllers:
        rospy.logwarn(f"No controller configuration found for {robot_name}; skipping controller spawner")
        return

    rosparam.upload_params(f"/{robot_name}", controllers)
    controller_names = list(controllers.keys())
    if not controller_names:
        rospy.logwarn(f"Controller list for {robot_name} is empty")
        return

    service_name = f"/{robot_name}/controller_manager/load_controller"
    try:
        rospy.wait_for_service(service_name, timeout=5.0)
    except rospy.ROSException:
        rospy.logwarn(f"Controller manager service for {robot_name} not available; skipping controllers")
        return

    cmd = [
        'rosrun',
        'controller_manager',
        'spawner',
        *controller_names,
        '--namespace',
        f'/{robot_name}'
    ]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        rospy.logerr("Controller spawner failed for %s: %s", robot_name, exc)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
