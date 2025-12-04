from __future__ import annotations

from typing import Optional

import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

from gazebo_rl_gym.envs.base.robot_spec import RobotEnvSpec


class ForkliftSpec(RobotEnvSpec):
    """Forklift spec with goal-conditioned reward similar to Turtlebot3."""

    POSE_KEYS: tuple[str, ...] = ("x", "y", "yaw", "vx", "wz")
    REWARD_POSITION_KEYS: tuple[str, ...] = ("x", "y")
    SCAN_DIM: int = 720

    def __init__(self, name: str, cfg):
        super().__init__(name, cfg)
        self._current_height: float = 0.0
        self._last_scan_min_value: Optional[float] = None

        task_cfg = getattr(cfg, "task")
        if not hasattr(task_cfg, "target_pos"):
            raise ValueError("Forklift robot requires 'task.target_pos' in configuration")

        target = getattr(task_cfg, "target_pos")
        if len(target) != 3:
            raise ValueError("task.target_pos must be a 3-element sequence [x, y, yaw]")

        self.target_x = float(target[0])
        self.target_y = float(target[1])
        self.target_yaw = float(target[2])

        self._prev_dist: Optional[float] = None
        self._curr_dist: Optional[float] = None
        self._curr_yaw_err_to_target: Optional[float] = None
        self._curr_vx: Optional[float] = None
        self._curr_wz: Optional[float] = None
        self._success_counter: int = 0

    def reset(self) -> None:
        super().reset()
        self._current_height = 0.0
        self._last_scan_min_value = None
        self._prev_dist = None
        self._curr_dist = None
        self._curr_yaw_err_to_target = None
        self._curr_vx = None
        self._curr_wz = None
        self._success_counter = 0

    @property
    def observation_dim(self) -> int:
        """Return observation dimension (pose + scan + goal features)."""
        base_dim = len(self.POSE_KEYS) + self.SCAN_DIM
        return base_dim + 3  # distance, relative bearing, yaw error

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
        pose_components = [mapping[key] for key in self.POSE_KEYS]
        pose_vec = np.array(pose_components, dtype=np.float32)

        if scan_msg is not None:
            scan = np.array(scan_msg.ranges, dtype=np.float32)
            scan = np.nan_to_num(scan, nan=scan_msg.range_max, posinf=scan_msg.range_max, neginf=scan_msg.range_min)
            if scan.size > self.SCAN_DIM:
                scan = scan[: self.SCAN_DIM]
            elif scan.size < self.SCAN_DIM:
                pad_val = scan_msg.range_max if scan_msg.range_max > 0.0 else 0.0
                scan = np.pad(scan, (0, self.SCAN_DIM - scan.size), mode="constant", constant_values=pad_val)
            self._last_scan_min_value = float(np.min(scan)) if scan.size else None

            normalize_scan = getattr(self.cfg, "observation", None) and getattr(self.cfg.observation, "normalize_scan")
            if normalize_scan and scan_msg.range_max > 0.0:
                scan = scan / scan_msg.range_max
        else:
            scan = np.ones(self.SCAN_DIM, dtype=np.float32)
            self._last_scan_min_value = None

        scan_vec = scan.astype(np.float32, copy=False)

        self._current_height = float(translation.z)

        dx = self.target_x - translation.x
        dy = self.target_y - translation.y
        dist = float(np.hypot(dx, dy))

        target_angle = np.arctan2(dy, dx) - yaw
        target_angle = (target_angle + np.pi) % (2 * np.pi) - np.pi

        yaw_err_to_target = self.target_yaw - yaw
        yaw_err_to_target = (yaw_err_to_target + np.pi) % (2 * np.pi) - np.pi

        self._curr_dist = dist
        self._curr_yaw_err_to_target = float(yaw_err_to_target)
        self._curr_vx = float(pose_msg.twist.twist.linear.x)
        self._curr_wz = float(pose_msg.twist.twist.angular.z)

        extra = np.array([dist, float(target_angle), float(yaw_err_to_target)], dtype=np.float32)

        return np.concatenate([pose_vec, scan_vec, extra])

    def compute_reward(self) -> float:
        reward_cfg = getattr(self.cfg, "reward")
        termination_cfg = getattr(self.cfg, "termination")
    
        curr = self.current_observation
        prev = self.previous_observation
        reward = 0.0
        reward_parts: dict[str, float] = {}

        # Projected displacement reward towards the goal
        pose_dim = len(self.POSE_KEYS)
        curr_pose = curr[:pose_dim]
        prev_pose = prev[:pose_dim]

        dx_move = float(curr_pose[0] - prev_pose[0])
        dy_move = float(curr_pose[1] - prev_pose[1])

        dx_target = self.target_x - float(prev_pose[0])
        dy_target = self.target_y - float(prev_pose[1])
        dist_to_target = float(np.hypot(dx_target, dy_target)) + 1e-8

        unit_to_goal = np.array([dx_target, dy_target], dtype=np.float32) / dist_to_target
        displacement_projection = float(np.dot(np.array([dx_move, dy_move], dtype=np.float32), unit_to_goal))
        distance_scale = float(getattr(reward_cfg, "distance_scale"))
        proj_reward = displacement_projection * distance_scale
        reward += proj_reward
        reward_parts["proj_displacement"] = proj_reward

        # Action smoothness reward
        smooth_scale = float(getattr(reward_cfg, "smoothness_scale"))
        smooth_penalty = 0.0
        if smooth_scale != 0.0 and self.current_action is not None and self.previous_action is not None:
            diff = self.current_action - self.previous_action
            smooth_penalty = float(np.dot(diff, diff)) * smooth_scale
            reward += smooth_penalty
        reward_parts["smoothness"] = smooth_penalty

        # Heading reward to encourage facing the goal
        heading_penalty_scale = float(getattr(reward_cfg, "heading_penalty_scale"))
        heading_penalty = 0.0
        if heading_penalty_scale != 0.0:
            curr_x = float(curr_pose[0])
            curr_y = float(curr_pose[1])
            target_heading = float(np.arctan2(self.target_y - curr_y, self.target_x - curr_x))
            robot_heading = float(curr_pose[2])
            heading_diff = (target_heading - robot_heading + np.pi) % (2 * np.pi) - np.pi
            heading_penalty = heading_penalty_scale * abs(heading_diff)
            reward += heading_penalty
        reward_parts["heading"] = heading_penalty

        # Collision reward 
        min_collision_range = getattr(termination_cfg, "min_collision_range")
        collision_penalty_cfg = float(getattr(reward_cfg, "collision_penalty"))
        collision_penalty = 0.0
        if (
            min_collision_range is not None
            and self._last_scan_min_value is not None
            and self._last_scan_min_value <= float(min_collision_range)
        ):
            collision_penalty = collision_penalty_cfg
            reward += collision_penalty
        reward_parts["collision"] = collision_penalty

        # Minimum lidar reward 
        min_range_penalty_cfg = float(getattr(reward_cfg, "min_range_penalty"))
        min_range_threshold = float(getattr(reward_cfg, "min_range_threshold"))
        min_range_penalty = 0.0
        if (
            self._last_scan_min_value is not None
            and min_range_threshold > 0.0
            and self._last_scan_min_value <= min_range_threshold
        ):
            shortfall = min_range_threshold - self._last_scan_min_value
            ratio = shortfall / max(min_range_threshold, 1e-6)
            min_range_penalty = min_range_penalty_cfg * ratio
            reward += min_range_penalty
        reward_parts["min_range"] = min_range_penalty

        # Distance shortening reward 朝着目标前进奖励
        distance_shortening_reward = 0.0
        dist_scale = float(getattr(reward_cfg, "dist_reward_scale"))
        if self._prev_dist is not None and self._curr_dist is not None:
            distance_shortening_reward = (self._prev_dist - self._curr_dist) * dist_scale
            reward += distance_shortening_reward
        reward_parts["distance_shortening"] = distance_shortening_reward
        self._prev_dist = self._curr_dist

        # Goal achievement reward
        goal_reward_value = 0.0
        success_pos = float(getattr(termination_cfg, "success_pos"))
        success_yaw = float(getattr(termination_cfg, "success_yaw"))
        success_lin_vel_th = float(getattr(termination_cfg, "success_lin_vel_th"))
        success_ang_vel_th = float(getattr(termination_cfg, "success_ang_vel_th"))

        pos_ok = self._curr_dist is not None and self._curr_dist < success_pos
        yaw_ok = (
            self._curr_yaw_err_to_target is not None
            and abs(self._curr_yaw_err_to_target) < success_yaw
        )
        vel_ok = (
            self._curr_vx is not None
            and self._curr_wz is not None
            and abs(self._curr_vx) < success_lin_vel_th
            and abs(self._curr_wz) < success_ang_vel_th
        )

        # position and orientation both achieved
        if pos_ok and yaw_ok and vel_ok:
            goal_reward_value = float(getattr(reward_cfg, "goal_reward"))
            reward += goal_reward_value
        
        # only achieved position
        if pos_ok:
            reward += goal_reward_value * 0.5 
        reward_parts["goal_reward"] = goal_reward_value

        # Pose-based reward within a gate radius 距离终点小于pose_gate_radius时计算,鼓励调整朝向
        pose_gate_radius = float(getattr(reward_cfg, "pose_gate_radius"))
        yaw_reward_scale = float(getattr(reward_cfg, "yaw_reward_scale"))
        pose_reward = 0.0
        if (
            pose_gate_radius > 0.0
            and yaw_reward_scale != 0.0
            and self._curr_dist is not None
            and self._curr_dist < pose_gate_radius
            and self._curr_yaw_err_to_target is not None
        ):
            yaw_err = self._curr_yaw_err_to_target
            pose_reward = yaw_reward_scale * (np.exp(abs(yaw_err)) - 1.0)
            reward += pose_reward
        reward_parts["pose"] = pose_reward

        step_cost = float(getattr(reward_cfg, "step_cost"))
        reward += step_cost
        reward_parts["step"] = step_cost

        if getattr(self.cfg, "reward_debug", False):
            print("----- Reward Debug Info -----")
            print(f"Reward breakdown ({self.name}): {reward_parts}")
            print(f"Lidar min distance: {self._last_scan_min_value}")
            print(f"Total reward: {reward}")

        return reward

    def is_done(self) -> bool:
        done = False

        # minimun distance collision done condition
        termination_cfg = getattr(self.cfg, "termination")
        collision_enabled = bool(getattr(termination_cfg, "collision"))
        min_collision_range = getattr(termination_cfg, "min_collision_range")
        if (
            collision_enabled
            and min_collision_range is not None
            and self._last_scan_min_value is not None
            and self._last_scan_min_value < float(min_collision_range)
        ):
            done |= True

        # height done condition
        min_height = getattr(termination_cfg, "min_height")
        if min_height is not None and self._current_height < float(min_height):
            done |= True

        # success pose stay done condition
        success_pos = float(getattr(termination_cfg, "success_pos"))
        success_yaw = float(getattr(termination_cfg, "success_yaw"))
        success_lin_vel_th = float(getattr(termination_cfg, "success_lin_vel_th"))
        success_ang_vel_th = float(getattr(termination_cfg, "success_ang_vel_th"))
        success_stay_steps = int(getattr(termination_cfg, "success_stay_steps"))

        pos_ok = self._curr_dist is not None and self._curr_dist < success_pos
        yaw_ok = (
            self._curr_yaw_err_to_target is not None
            and abs(self._curr_yaw_err_to_target) < success_yaw
        )
        vel_ok = (
            self._curr_vx is not None
            and self._curr_wz is not None
            and abs(self._curr_vx) < success_lin_vel_th
            and abs(self._curr_wz) < success_ang_vel_th
        )

        if pos_ok and yaw_ok and vel_ok:
            self._success_counter += 1
        else:
            self._success_counter = 0
        done |= (self._success_counter >= success_stay_steps)
        
        if getattr(self.cfg, "reward_debug"):
            print("----- Termination Debug Info -----")
            print(f"Success counter: {self._success_counter}/{success_stay_steps}")
            print(f"Done: {done}")

        return done


        

    def command_from_action(self, action: np.ndarray) -> Twist:
        twist = Twist()
        twist.linear.x = float(np.clip(action[0], self.action_low[0], self.action_high[0]))
        twist.angular.z = float(np.clip(action[1], self.action_low[1], self.action_high[1]))
        return twist