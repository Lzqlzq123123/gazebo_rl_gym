from __future__ import annotations

from typing import Optional

import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

from gazebo_rl_gym.envs.base.robot_spec import RobotEnvSpec


class NanocarSpec(RobotEnvSpec):
    """Concrete NanoCar spec duplicating diff-drive default behaviour."""

    POSE_KEYS: tuple[str, ...] = ("x", "y", "yaw")
    REWARD_POSITION_KEYS: tuple[str, ...] = ("x", "y")
    SCAN_DIM: int = 720

    def __init__(self, name: str, cfg):
        super().__init__(name, cfg)
        self._current_height: float = 0.0

    def reset(self) -> None:
        super().reset()
        self._current_height = 0.0

    @property
    def observation_dim(self) -> int:
        """
        Return the dimension of the observation vector produced by this spec.
        """
        return len(self.POSE_KEYS) + self.SCAN_DIM

    @property
    def uses_scan(self) -> bool:
        return self.SCAN_DIM > 0 and bool(super().uses_scan)

    def build_observation(self, pose_msg: Odometry, scan_msg: Optional[LaserScan]) -> np.ndarray:
        translation = pose_msg.pose.pose.position
        rotation = pose_msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([rotation.x, rotation.y, rotation.z, rotation.w])

        mapping = {
            "x": translation.x,
            "y": translation.y,
            "z": translation.z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "vx": pose_msg.twist.twist.linear.x,
            "vy": pose_msg.twist.twist.linear.y,
            "vz": pose_msg.twist.twist.linear.z,
            "wx": pose_msg.twist.twist.angular.x,
            "wy": pose_msg.twist.twist.angular.y,
            "wz": pose_msg.twist.twist.angular.z,
        }
        components: list[float] = []
        for key in self.POSE_KEYS:
            if key not in mapping:
                raise KeyError(f"Unsupported pose component '{key}' for robot '{self.name}'")
            components.append(mapping[key])
        pose_vec = np.array(components, dtype=np.float32) if components else np.empty(0, dtype=np.float32)

        scan_vec = np.empty(0, dtype=np.float32)
        if self.SCAN_DIM > 0 and scan_msg is not None:
            scan = np.array(scan_msg.ranges, dtype=np.float32)
            scan = np.nan_to_num(scan, nan=scan_msg.range_max, posinf=scan_msg.range_max, neginf=0.0)
            normalize_scan = getattr(self.cfg, "observation", None) and getattr(self.cfg.observation, "normalize_scan", True)
            if normalize_scan and scan_msg.range_max > 0:
                scan /= scan_msg.range_max

            if scan.size > self.SCAN_DIM:
                scan = scan[:self.SCAN_DIM]
            elif scan.size < self.SCAN_DIM:
                pad_val = 1.0 if normalize_scan else scan_msg.range_max or 0.0
                scan = np.pad(scan, (0, self.SCAN_DIM - scan.size), mode="constant", constant_values=pad_val)
            scan_vec = scan.astype(np.float32, copy=False)

        self._current_height = float(translation.z)

        if pose_vec.size and scan_vec.size:
            return np.concatenate([pose_vec, scan_vec])
        if pose_vec.size:
            return pose_vec
        if scan_vec.size:
            return scan_vec
        return np.empty((0,), dtype=np.float32)

    def compute_reward(self) -> float:
        curr = self.current_observation
        prev = self.previous_observation
        if curr is None or prev is None:
            return 0.0

        pose_dim = len(self.POSE_KEYS)
        if pose_dim == 0:
            return 0.0

        curr_pose = curr[:pose_dim]
        prev_pose = prev[:pose_dim]
        indices = {comp: idx for idx, comp in enumerate(self.POSE_KEYS)}

        deltas = []
        for comp in self.REWARD_POSITION_KEYS:
            idx = indices.get(comp)
            if idx is not None:
                deltas.append(curr_pose[idx] - prev_pose[idx])

        distance_scale = getattr(self.cfg.reward, "distance_scale", 1.0)
        reward = float(np.linalg.norm(deltas) * distance_scale) if deltas else 0.0

        smooth_scale = getattr(self.cfg.reward, "smoothness_scale", 0.0)
        if smooth_scale > 0.0:
            curr_action = self.current_action
            prev_action = self.previous_action
            if curr_action is not None and prev_action is not None:
                diff = curr_action - prev_action
                reward -= float(smooth_scale * np.dot(diff, diff))

        return reward

    def is_done(self) -> bool:
        min_height = getattr(self.cfg.termination, "min_height", None)
        if min_height is None:
            return False
        return bool(self._current_height < float(min_height))

    def command_from_action(self, action: np.ndarray) -> Twist:
        if action.size < 2:
            raise ValueError(f"Action for robot '{self.name}' must have at least 2 elements")

        twist = Twist()
        twist.linear.x = float(np.clip(action[0], self.action_low[0], self.action_high[0]))
        twist.angular.z = float(np.clip(action[1], self.action_low[1], self.action_high[1]))
        return twist

