from __future__ import annotations

"""Single-robot Gazebo environment (no gym dependency).

This class is responsible for:
- Loading an environment YAML (world + robot preset + initial pose).
- Launching Gazebo with the specified world.
- Creating a single RobotEnvSpec (e.g. Turtlebot3Spec) from the preset.
- Managing ROS topics (pose/scan/contact/cmd_vel).
- Providing a simple reset/step API returning numpy arrays.

It is intentionally kept minimal and focused on one RL-controlled robot.
Additional robots (e.g. IL/scripted agents) can be added later as part of the
same Gazebo simulation.
"""

from typing import Tuple, Dict, Any

import os
import time

import numpy as np
import roslaunch
import rospy
from gazebo_msgs.msg import ContactsState, ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from tf.transformations import quaternion_from_euler

from gazebo_rl_gym.envs.base.robot_spec import RobotEnvSpec
from gazebo_rl_gym.utils.config_loader import load_environment_config
from gazebo_rl_gym.utils.path_utils import resolve_relative


class GazeboSingleRobotEnv:
    """Single-robot Gazebo environment wrapping one RobotEnvSpec.

    目前实现为：从多机器人环境逻辑中抽取出“单机器人 + world”版本，
    用于和 rsl_rl 的 VecEnv 一起工作。
    """

    def __init__(self, env_cfg_path: str):
        # 加载环境配置（world + robots 列表），这里沿用多机器人配置格式，
        # 但只使用 robots[0] 作为 RL 控制的机器人。
        self.config_path = os.path.abspath(env_cfg_path)
        self.env_config = load_environment_config(self.config_path)

        # 解析机器人配置，当前只使用第一个机器人
        self.robot_configs = self.env_config.robots
        if not self.robot_configs:
            raise ValueError(f"No robots defined in configuration '{self.config_path}'")
        self.robot_cfg = self.robot_configs[0]
        self.robot_name: str = self.robot_cfg.name

        # 这里 env_config 中的 robot_cfg.spec 已经是一个 RobotEnvSpec 实例
        spec = self.robot_cfg.spec
        if not isinstance(spec, RobotEnvSpec):
            raise TypeError("Expected robot_cfg.spec to be a RobotEnvSpec instance")
        self.spec: RobotEnvSpec = spec

        # 观测/动作维度直接来自 spec
        self._obs_dim: int = self.spec.observation_dim
        self._action_dim: int = self.spec.action_dim

        # Gazebo / ROS 状态
        self._physics_paused = False
        self.sim_time: float | None = None
        self._clock_event = rospy.Event() if hasattr(rospy, "Event") else None

        # 传感器缓存
        self.scan_buffer: dict[str, LaserScan | None] = {self.robot_name: None}
        self.pose_buffer: dict[str, Odometry | None] = {self.robot_name: None}
        self.scan_stamp: dict[str, float | None] = {self.robot_name: None}
        self.pose_stamp: dict[str, float | None] = {self.robot_name: None}
        self.contact_buffer: dict[str, bool] = {self.robot_name: False}

        # ROS topic 及 publisher/订阅
        self.action_pub: Twist | None = None  # type: ignore[assignment]

        # 时序与超时配置
        metadata = self.env_config.metadata or {}
        self.topic_timeout = float(metadata.get("topic_timeout", 2.0))
        self.sim_wait_timeout = float(metadata.get("sim_wait_timeout", max(self.topic_timeout, 3.0)))
        self.min_sim_dt = float(metadata.get("min_sim_dt", 0.01))
        self.model_wait_timeout = float(metadata.get("model_wait_timeout", 15.0))
        if self.sim_wait_timeout <= 0:
            raise ValueError("sim_wait_timeout must be positive")
        if self.min_sim_dt <= 0:
            raise ValueError("min_sim_dt must be positive")
        if self.model_wait_timeout <= 0:
            raise ValueError("model_wait_timeout must be positive")

        # 启动 Gazebo 并初始化 ROS
        self._launch_gazebo()
        rospy.init_node("gazebo_single_env", anonymous=True)

        # 订阅 clock
        rospy.Subscriber("/clock", Clock, self._clock_callback)

        # 建立 ROS service 句柄
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.reset_simulation = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        # 创建 ROS pub/sub
        self._setup_robot_interfaces()

        rospy.loginfo(
            "Configured single robot: %s (%s)",
            self.robot_name,
            self.spec.cfg.model,
        )

    # ------------------------------------------------------------------
    # 公共属性
    # ------------------------------------------------------------------
    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # ------------------------------------------------------------------
    # 核心 API：reset / step
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset Gazebo world and the robot, then return the initial observation."""
        # 调用 Gazebo reset 服务（注意：reset_world 会自动暂停物理引擎）
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_world()
        except rospy.ServiceException as exc:
            rospy.logwarn("reset_world failed: %s, falling back to reset_simulation", exc)
            rospy.wait_for_service("/gazebo/reset_simulation")
            self.reset_simulation()

        # reset 后物理引擎被暂停，标记状态
        self._physics_paused = True

        # 重置内部时间与缓存
        prev_sim_time = self.sim_time
        self.sim_time = None
        if self._clock_event is not None:
            self._clock_event.clear()

        # 等待模型加载并重置初始位姿
        self._wait_for_model()
        self._reset_robot_pose(self.robot_cfg)

        # 清空 sensor 缓存，重置 spec 状态
        self._clear_sensor_buffers()

        # 关键修复：立即 unpause 让时钟开始运行
        self._unpause_physics(force=True)
        
        # 等待时钟开始运行（从 reset 后的状态恢复）
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if self.sim_time is not None and (prev_sim_time is None or self.sim_time != prev_sim_time):
                break
            time.sleep(0.01)
        
        if self.sim_time is None:
            rospy.logwarn("Clock not advancing after reset, forcing unpause again...")
            self._unpause_physics(force=True)
            time.sleep(0.5)

        # 推动仿真，直到收到新的传感器数据
        self._advance_simulation(
            min_dt=self.min_sim_dt,
            timeout=self.sim_wait_timeout,
            require_fresh_sensors=True,
            prev_scan_stamp=None,
            prev_pose_stamp=None,
        )

        # 构造初始观测
        obs = self._get_observation()
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply an action and step the simulation.

        Returns (next_obs, reward, done, info).
        """
        # 先 pause，保证动作是确定性应用
        self._pause_physics()

        prev_scan_stamp = self.scan_stamp[self.robot_name]
        prev_pose_stamp = self.pose_stamp[self.robot_name]

        # 发布动作
        action_vec = np.asarray(action, dtype=np.float32)
        twist = self.spec.apply_action(action_vec)
        self.action_pub.publish(twist)

        # 推动仿真，直到传感器数据更新
        self._advance_simulation(
            min_dt=self.min_sim_dt,
            timeout=self.sim_wait_timeout,
            require_fresh_sensors=True,
            prev_scan_stamp=prev_scan_stamp,
            prev_pose_stamp=prev_pose_stamp,
        )

        # 采集观测、奖励、终止信息
        obs = self._get_observation()
        reward = float(self.spec.compute_reward())
        done = bool(self.spec.is_done())

        info: Dict[str, Any] = {}
        if done:
            # 可以在这里填 episode 统计（长度、累计奖励等），后续再加
            info["episode"] = {}
            obs = self.reset()
        return obs, reward, done, info

    def close(self) -> None:
        """Clean up resources (e.g. stop roslaunch)."""
        if hasattr(self, "launch"):
            self.launch.shutdown()

    # ------------------------------------------------------------------
    # 内部工具：Gazebo / ROS 控制
    # ------------------------------------------------------------------
    def _clock_callback(self, msg: Clock) -> None:
        self.sim_time = msg.clock.to_sec()
        if self._clock_event is not None:
            self._clock_event.set()

    def _pause_physics(self, *, force: bool = False) -> None:
        if getattr(self, "_physics_paused", False) and not force:
            return
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as exc:
            rospy.logerr("pause_physics failed: %s", exc)
            raise
        self._physics_paused = True

    def _unpause_physics(self, *, force: bool = False) -> None:
        if not getattr(self, "_physics_paused", False) and not force:
            return
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as exc:
            rospy.logerr("unpause_physics failed: %s", exc)
            raise
        self._physics_paused = False

    def _advance_simulation(
        self,
        *,
        min_dt: float,
        timeout: float,
        require_fresh_sensors: bool = False,
        prev_scan_stamp: float | None = None,
        prev_pose_stamp: float | None = None,
    ) -> None:
        """Unpause Gazebo, wait for the clock/sensors to advance, then pause again."""
        start_sim_time = self.sim_time
        deadline = time.time() + timeout

        self._unpause_physics(force=True)
        try:
            while time.time() < deadline:
                current_sim_time = self.sim_time
                if start_sim_time is None and current_sim_time is not None:
                    start_sim_time = current_sim_time

                sim_ready = (
                    current_sim_time is not None
                    and start_sim_time is not None
                    and current_sim_time >= start_sim_time + min_dt - 1e-9
                )

                sensors_ready = True
                if require_fresh_sensors:
                    sensors_ready = False
                    pose_stamp = self.pose_stamp[self.robot_name]
                    scan_stamp = self.scan_stamp[self.robot_name]

                    if pose_stamp is None:
                        sensors_ready = False
                    elif prev_pose_stamp is not None and pose_stamp <= prev_pose_stamp:
                        sensors_ready = False
                    else:
                        if self.spec.uses_scan:
                            if scan_stamp is None:
                                sensors_ready = False
                            elif prev_scan_stamp is not None and scan_stamp <= prev_scan_stamp:
                                sensors_ready = False
                            else:
                                sensors_ready = True
                        else:
                            sensors_ready = True

                if sim_ready and sensors_ready:
                    return

                time.sleep(0.001)

            rospy.logerr("Timed out waiting for simulation advance (Δt>=%.3fs)", min_dt)
            raise RuntimeError("Timed out waiting for simulation advance")
        finally:
            self._pause_physics(force=True)

    def _scan_callback(self, msg: LaserScan) -> None:
        self.scan_buffer[self.robot_name] = msg
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        self.scan_stamp[self.robot_name] = stamp

    def _pose_callback(self, msg: Odometry) -> None:
        self.pose_buffer[self.robot_name] = msg
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        self.pose_stamp[self.robot_name] = stamp

    def _contact_callback(self, msg: ContactsState) -> None:
        collision = bool(msg.states)
        self.contact_buffer[self.robot_name] = collision
        self.spec.update_contact_state(msg)

    def _setup_robot_interfaces(self) -> None:
        # Publisher for cmd_vel
        self.action_pub = rospy.Publisher(self.spec.cmd_topic, Twist, queue_size=1)

        # Pose subscriber
        rospy.Subscriber(self.spec.pose_topic, Odometry, self._pose_callback)

        # Lidar subscriber (if used)
        if self.spec.uses_scan:
            rospy.Subscriber(self.spec.scan_topic, LaserScan, self._scan_callback)

        # Contact subscriber (if used)
        if self.spec.uses_contact:
            rospy.Subscriber(self.spec.contact_topic, ContactsState, self._contact_callback)

    def _wait_for_model(self) -> None:
        wait_timeout = self.model_wait_timeout
        if wait_timeout <= 0:
            return

        pending = {self.robot_name}
        deadline = time.time() + wait_timeout
        last_log = 0.0

        while pending and time.time() < deadline:
            try:
                response = self.get_model_state(self.robot_name, "")
            except rospy.ServiceException as exc:
                rospy.logdebug("get_model_state for %s failed: %s", self.robot_name, exc)
                continue

            if response.success:
                pending.clear()
                break

            now = time.time()
            if now - last_log > 1.0:
                rospy.loginfo("Waiting for model to spawn: %s", self.robot_name)
                last_log = now
            time.sleep(0.1)

        if pending:
            raise RuntimeError(
                f"Timed out waiting for model to spawn: {self.robot_name}. check spawner logs."
            )

        rospy.loginfo("Detected model in Gazebo: %s", self.robot_name)

    def _clear_sensor_buffers(self) -> None:
        self.scan_buffer[self.robot_name] = None
        self.pose_buffer[self.robot_name] = None
        self.scan_stamp[self.robot_name] = None
        self.pose_stamp[self.robot_name] = None
        self.contact_buffer[self.robot_name] = False
        self.spec.reset()
        if self.spec.uses_contact:
            self.spec.update_contact_state(None)

    def _reset_robot_pose(self, robot_config) -> None:
        state_msg = ModelState()
        state_msg.model_name = robot_config.name

        pose = robot_config.pose
        state_msg.pose.position.x = pose.get("x", 0.0)
        state_msg.pose.position.y = pose.get("y", 0.0)
        state_msg.pose.position.z = pose.get("z", 0.0)
        q = quaternion_from_euler(0, 0, pose.get("yaw", 0.0))
        state_msg.pose.orientation.x = q[0]
        state_msg.pose.orientation.y = q[1]
        state_msg.pose.orientation.z = q[2]
        state_msg.pose.orientation.w = q[3]

        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            self.set_model_state(state_msg)
        except rospy.ServiceException as exc:
            rospy.logerr("set_model_state failed: %s", exc)

    def _get_observation(self) -> np.ndarray:
        """Block until we have fresh pose/scan, then build observation via spec."""
        name = self.robot_name
        spec = self.spec

        # 等待必要的传感器数据
        if self.pose_buffer[name] is None or (spec.uses_scan and self.scan_buffer[name] is None):
            rospy.logwarn(f"Waiting for initial sensor data for {name}")
            start = time.time()
            while self.pose_buffer[name] is None or (spec.uses_scan and self.scan_buffer[name] is None):
                if time.time() - start > self.topic_timeout:
                    missing: list[str] = []
                    if self.pose_buffer[name] is None:
                        missing.append("pose")
                    if spec.uses_scan and self.scan_buffer[name] is None:
                        missing.append("scan")
                    rospy.logerr(
                        "Missing %s for %s after %.1fs", ",".join(missing), name, self.topic_timeout
                    )
                    raise RuntimeError(f"Timed out waiting for observation data for robot {name}")
                time.sleep(0.01)

        pose_msg = self.pose_buffer[name]
        scan_msg = self.scan_buffer[name] if spec.uses_scan else None
        assert pose_msg is not None
        obs = spec.process_sensor_data(pose_msg, scan_msg)
        return obs

    def _launch_gazebo(self) -> None:
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)

        launch_file = resolve_relative("launch", "multi_robot.launch")

        launch_args = [
            f"config_file:={self.config_path}",
            f"world_file:={self.env_config.world_file}",
        ]

        print(f"Launch args: {launch_args}")
        print(f"World file: {self.env_config.world_file}")

        self.launch = roslaunch.parent.ROSLaunchParent(uuid, [(launch_file, launch_args)])
        self.launch.start()
        print("Gazebo launched")
