from __future__ import annotations

from typing import Optional

# matplotlib.pyplot.sca was imported but not used; remove to avoid lint issues
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

from gazebo_rl_gym.envs.base.robot_spec import RobotEnvSpec


class Turtlebot3Spec(RobotEnvSpec):
    """Concrete TurtleBot3 spec with diff-drive control and lidar sensing."""

    POSE_KEYS: tuple[str, ...] = ("x", "y", "yaw","vx", "vy", "wz")
    REWARD_POSITION_KEYS: tuple[str, ...] = ("x", "y")
    SCAN_DIM: int = 360

    def __init__(self, name: str, cfg):
        super().__init__(name, cfg)
        self._current_height: float = 0.0
        self._last_scan_min_value: Optional[float] = None

        # 目标点导航相关配置
        task_cfg = cfg.task
        target = task_cfg.target_pos
    
        self.target_x = float(target[0])
        self.target_y = float(target[1])
        self.target_yaw = float(target[2])
        self.success_radius = float(getattr(task_cfg, "success_radius", 0.3))
        print(f"Target position: x={self.target_x}, y={self.target_y}, yaw={self.target_yaw}")

        # 上一帧与目标距离，用于距离缩短奖励与终点判定
        self._prev_dist: Optional[float] = None

    def reset(self) -> None:
        super().reset()
        self._current_height = 0.0
        self._last_scan_min_value = None

    @property
    def observation_dim(self) -> int:
        """
        Return the dimension of the observation vector produced by this spec.
        """
        # 基础观测：位姿 + 激光
        base_dim = len(self.POSE_KEYS) + self.SCAN_DIM
        # 额外添加 2 维：与目标点的距离 + 目标在机器人坐标系中的角度
        return base_dim + 2

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
            components.append(mapping[key])
        pose_vec = np.array(components, dtype=np.float32) 

        self._last_scan_min_value = None 
        scan = np.array(scan_msg.ranges, dtype=np.float32)
        # 无效 / 无穷点都被当成“很远的障碍”
        scan = np.nan_to_num(
            scan,
            nan=scan_msg.range_max,
            posinf=scan_msg.range_max,
            neginf=scan_msg.range_max,
        )

        if scan.size > self.SCAN_DIM:
            scan = scan[:self.SCAN_DIM]
        self._last_scan_min_value = float(np.min(scan))

        if getattr(self.cfg, "observation", None) and getattr(self.cfg.observation, "normalize_scan", True) and scan_msg.range_max > 0:
            scan /= scan_msg.range_max

        self._current_height = float(translation.z)

        # 与目标点的距离和相对角度（在机器人坐标系中）
        dx = self.target_x - translation.x
        dy = self.target_y - translation.y
        dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)

        target_angle = np.arctan2(dy, dx) - yaw
        target_angle = (target_angle + np.pi) % (2 * np.pi) - np.pi
        target_angle = target_angle.astype(np.float32)

        extra = np.array([dist, target_angle], dtype=np.float32)

        parts = [vec for vec in (pose_vec, scan, extra) if vec.size]
        return np.concatenate(parts)
    
    def compute_reward(self) -> float:
        reward = 0.0
        reward_parts = {}  # 用于存储各部分奖励
        
        # 获取当前和前一时刻的观测值
        curr = self.current_observation
        prev = self.previous_observation
        # 获取姿态维度
        pose_dim = len(self.POSE_KEYS)
        # 提取当前和前一时刻的姿态部分
        curr_pose = curr[:pose_dim]
        prev_pose = prev[:pose_dim]
        # 创建姿态组件到索引的映射
        indices = {comp: idx for idx, comp in enumerate(self.POSE_KEYS)}
        # 计算奖励位置键的增量
        deltas = []
        for comp in self.REWARD_POSITION_KEYS:
            idx = indices.get(comp)
            if idx is not None:
                deltas.append(curr_pose[idx] - prev_pose[idx])
        
        # 计算基础位移奖励（投影方式）：只对沿着目标方向的分量进行奖励或惩罚（有符号）
        # ★ 修复：只使用位置增量（x, y），不混入角度增量，因为量纲不同
        distance_scale = float(getattr(self.cfg.reward, "distance_scale"))
        
        # 当前位置相对上一帧的位移向量（仅 x, y 坐标）
        dx_move = curr_pose[0] - prev_pose[0]
        dy_move = curr_pose[1] - prev_pose[1]
        
        # 从上一帧位置指向目标的向量（仅 x, y 坐标）
        dx_target = self.target_x - prev_pose[0]
        dy_target = self.target_y - prev_pose[1]
        dist_to_target = float(np.sqrt(dx_target**2 + dy_target**2)) + 1e-8
        
        # 朝目标的单位向量
        unit_to_goal_xy = np.array([dx_target, dy_target], dtype=np.float32) / dist_to_target
        
        # 投影奖励（只用位置，不用角度）
        proj_xy = float(np.dot(np.array([dx_move, dy_move], dtype=np.float32), unit_to_goal_xy))
        proj_reward = proj_xy * distance_scale
        reward += proj_reward
        reward_parts["proj_displacement"] = proj_reward

        # 计算动作平滑惩罚
        smooth_scale = float(getattr(self.cfg.reward, "smoothness_scale"))
        smoothness_penalty = 0.0
        if smooth_scale != 0.0 and self.previous_action is not None:
            diff = self.current_action - self.previous_action
            smoothness_penalty =float(np.dot(diff, diff)) * smooth_scale
            reward += smoothness_penalty
        reward_parts["smoothness"] = smoothness_penalty
        
        # ★ 新增：方向惩罚项（可选），防止不必要的旋转
        # 如果启用，会惩罚与目标方向不一致的机器人朝向
        heading_penalty_scale = float(getattr(self.cfg.reward, "heading_penalty_scale", 0.0))
        heading_penalty_value = 0.0
        if heading_penalty_scale != 0.0:
            # 目标方向（从当前位置指向目标）
            curr_x, curr_y = float(curr_pose[0]), float(curr_pose[1])
            dx_to_target = self.target_x - curr_x
            dy_to_target = self.target_y - curr_y
            target_heading = float(np.arctan2(dy_to_target, dx_to_target))
            
            # 机器人当前朝向
            robot_heading = float(curr_pose[2])
            
            # 计算方向差（归一化到 [-π, π]）
            heading_diff = target_heading - robot_heading
            heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
            
            # 方向偏差越大惩罚越大
            heading_penalty_value = heading_penalty_scale * abs(float(heading_diff))
            reward += heading_penalty_value
        reward_parts["heading"] = heading_penalty_value


        # 基于激光的碰撞等效惩罚与避障惩罚
        min_collision_range = getattr(self.cfg.termination, "min_collision_range")
        collision_penalty = self.cfg.reward.collision_penalty
        min_range_penalty = self.cfg.reward.min_range_penalty
        min_range_threshold = self.cfg.reward.min_range_threshold

        # 碰撞等效惩罚：最小距离小于碰撞阈值
        collision_penalty_value = 0.0
        if self._last_scan_min_value <= min_collision_range:
            collision_penalty_value = collision_penalty
            reward += collision_penalty_value
        reward_parts["collision"] = collision_penalty_value

        # 最小范围惩罚（基于激光扫描）用于避障
        min_range_penalty_value = 0.0
        if self._last_scan_min_value <= min_range_threshold:
            shortfall = min_range_threshold - self._last_scan_min_value
            base_shortfall = float(shortfall / max(min_range_threshold, 1e-6))
            min_range_penalty_value = min_range_penalty * base_shortfall
            reward += min_range_penalty_value
        reward_parts["min_range"] = min_range_penalty_value

        # ---------------------------
        # 目标点导航相关奖励：距离缩短 + 终点奖励 + 每步代价
        # ---------------------------
        pose_dim = len(self.POSE_KEYS)
        curr_pose = curr[:pose_dim]
        x, y = float(curr_pose[0]), float(curr_pose[1])


        dx = self.target_x - x
        dy = self.target_y - y
        dist = float(np.sqrt(dx * dx + dy * dy))

        # 距离缩短奖励
        distance_shortening_reward = 0.0
        if self._prev_dist is not None:
            delta = self._prev_dist - dist  # 距离减小为正
            dist_scale = getattr(self.cfg.reward, "dist_reward_scale")
            distance_shortening_reward = dist_scale * delta
            reward += distance_shortening_reward
        reward_parts["distance_shortening"] = distance_shortening_reward
        self._prev_dist = dist

        # 到达终点的一次性奖励
        goal_reward_value = 0.0
        if dist < self.success_radius:
            goal_reward = float(getattr(self.cfg.reward, "goal_reward"))
            goal_reward_value = goal_reward
            reward += goal_reward_value
        reward_parts["goal_reward"] = goal_reward_value

        # 每步时间成本（负奖励，鼓励尽快到达目标）
        step_cost = float(getattr(self.cfg.reward, "step_cost"))
        step_cost_value = step_cost
        reward += step_cost_value
        reward_parts["step"] = step_cost_value

        # 打印详细的奖励信息（可通过 cfg 中的 reward_debug 开关打开）
        if getattr(self.cfg, 'reward_debug'):
            print(f"Reward breakdown: {reward_parts}")
            print(f"Total reward: {reward}")
            print("curr_pose:", curr_pose)
            print(f"Min scan value: {self._last_scan_min_value:.3f}")
            print("---")

        return reward

    def is_done(self) -> bool:
        # 基于激光的碰撞等效终止
        min_collision_range = getattr(self.cfg.termination, "min_collision_range")
        # print( "Min scan value:", self._last_scan_min_value)
        # print("bool",self._last_scan_min_value<= min_collision_range)
        if self._last_scan_min_value is not None and self._last_scan_min_value < min_collision_range:
            return True

        # 高度终止（掉下台等）
        min_height = getattr(self.cfg.termination, "min_height", None)
        if min_height is not None and self._current_height < float(min_height):
            return True

        # 到达目标点终止
        if self._prev_dist is not None and self._prev_dist < self.success_radius:
            return True

        return False

    def command_from_action(self, action: np.ndarray) -> Twist:
        twist = Twist()
        twist.linear.x = float(np.clip(action[0], self.action_low[0], self.action_high[0]))
        twist.angular.z = float(np.clip(action[1], self.action_low[1], self.action_high[1]))
        return twist

