# Gazebo RL Gym（单车 + rsl_rl 版本）

当前版本主要支持**单车无人车导航任务**，使用 ROS + Gazebo 作为仿真环境，底层用 `Turtlebot3Spec` 定义观测 / 奖励 / 终止逻辑，上层通过 `GazeboSingleRobotEnv` + `GazeboSingleVecEnv` 适配成 `rsl_rl` 的 `VecEnv` 接口，训练由 `rsl_rl.algorithms.PPO` + `rsl_rl.runners.OnPolicyRunner` 完成。

旧版自实现 PPO 与多机器人训练入口已移除，推荐统一使用 `train_rsl_ppo_single.py` + YAML 配置进行训练。

## 代码结构（简版）

- `config/`
    - `config/envs/warehouse.yaml`：环境配置（world + 机器人初始位姿 + 任务 / 奖励 / 终止参数）。
    - `config/train/warehouse_ppo_turtlebot3.yaml`：训练配置（PPO 超参、网络结构、采样步数等）。
- `scripts/gazebo_rl_gym/envs/`
    - `turtlebot3/turtlebot3_env.py`：`Turtlebot3Spec`，定义无人车任务的观测、奖励和终止条件。
    - `gazebo_single_env.py`：`GazeboSingleRobotEnv`，单车 Gazebo 封装，负责 ROS 话题和 Gazebo 服务交互。
    - `gazebo_vec_env.py`：`GazeboSingleVecEnv`，将单车环境封装为 `rsl_rl.env.VecEnv`。
- `scripts/train_rsl_ppo_single.py`：训练入口，读取 YAML，构建 VecEnv 和 OnPolicyRunner 并启动训练。

## 从“设计”到“被模型用上”：观测 & 奖励数据流

这一节以 TurtleBot3 导航为例，从底层设计到 PPO 网络使用，梳理观测和奖励的完整路径，并用“添加一个额外观测 + 奖励项”的例子说明修改点。

### 1. 任务设计层：`Turtlebot3Spec`

文件：`scripts/gazebo_rl_gym/envs/turtlebot3/turtlebot3_env.py`

这里继承 `RobotEnvSpec`，主要负责：

- 定义观测布局和维度：
    - `POSE_KEYS`：位姿 / 速度分量，例如 `(x, y, yaw, vx, vy, wz)`。
    - `SCAN_DIM`：激光雷达维度，例如 `360`。
    - `observation_dim` 属性：通常是 `len(POSE_KEYS) + SCAN_DIM` 再加上你额外拼接的量。
- 构造观测：`build_observation(self, pose_msg, scan_msg) -> np.ndarray`
    - 从 `Odometry` 中取位姿、速度；从 `LaserScan` 中取 ranges；
    - 进行必要的归一化和裁剪；
    - 按 `POSE_KEYS` 顺序拼出位姿向量 `pose_vec`；
    - 将 `pose_vec` 和 `scan` 拼接为最终观测向量 `obs`（`np.ndarray`）。
- 计算奖励：`compute_reward(self) -> float`
    - 使用当前 / 前一帧观测（`self.current_observation` / `self.previous_observation`）和动作（`self.current_action` / `self.previous_action`）计算奖励；
    - 示例：
        - 位移奖励：当前位置与上一位置的增量（`distance_scale`）。
        - 动作平滑惩罚：当前动作与前一动作差的平方（`smoothness_scale`）。
        - 碰撞惩罚：`collision_penalty`。
        - 激光最小距离惩罚：`min_range_penalty`，阈值 `min_range_threshold`。
- 终止条件：`is_done(self) -> bool`
    - 示例：碰撞标志 + 高度阈值 `min_height`。

**例子：添加一个额外观测 + 奖励项**

假设要增加“机器人与目标点的欧氏距离”作为观测，并根据距离缩短给予奖励：

1. 在环境 YAML（例如 `config/envs/warehouse.yaml`）中的 robot 配置里添加任务参数：

     ```yaml
     robots:
         - name: turtle1
             preset: turtlebot3_waffle
             pose: {x: 0.0, y: 0.0, z: 0.1, yaw: 0.0}
             task:
                 target_pos: [2.0, 1.0]       # 目标点 (x, y)
             reward:
                 dist_reward_scale: 1.0       # 距离缩短奖励系数
     ```

2. 在 `Turtlebot3Spec.__init__` 中读取：

     ```python
     class Turtlebot3Spec(RobotEnvSpec):
             def __init__(self, name: str, cfg):
                     super().__init__(name, cfg)
                     task_cfg = getattr(cfg, "task", None) or {}
                     target = getattr(task_cfg, "target_pos", [1.0, 0.0])
                     self.target_x = float(target[0])
                     self.target_y = float(target[1])
                     self.prev_dist: float | None = None
     ```

3. 在 `build_observation` 里追加距离：

     ```python
     def build_observation(self, pose_msg: Odometry, scan_msg: Optional[LaserScan]) -> np.ndarray:
             # ... 原有位姿 + 激光处理，得到 base_obs

             x = pose_msg.pose.pose.position.x
             y = pose_msg.pose.pose.position.y
             dx = self.target_x - x
             dy = self.target_y - y
             dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)

             obs = np.concatenate([base_obs, np.array([dist], dtype=np.float32)])
             return obs
     ```

4. 在 `compute_reward` 中加入距离缩短奖励：

     ```python
     def compute_reward(self) -> float:
             reward = 0.0
             # ... 原有位移、平滑、碰撞等项

             # 根据 current_observation / previous_observation 还原当前位置
             curr = self.current_observation
             pose_dim = len(self.POSE_KEYS)
             curr_pose = curr[:pose_dim]
             x, y = curr_pose[0], curr_pose[1]

             dx = self.target_x - x
             dy = self.target_y - y
             dist = float(np.sqrt(dx * dx + dy * dy))

             if self.prev_dist is not None:
                     delta = self.prev_dist - dist  # 距离减少为正
                     scale = getattr(self.cfg.reward, "dist_reward_scale", 1.0)
                     reward += scale * delta
             self.prev_dist = dist

             return reward
     ```

到这里，**观测向量多了一维距离；奖励里多了一项“距离缩短奖励”**，后续层会自动感知新的维度。

### 2. 仿真层：`GazeboSingleRobotEnv`

文件：`scripts/gazebo_rl_gym/envs/gazebo_single_env.py`

职责：

- 根据环境 YAML 加载 world 和机器人配置（`warehouse.yaml`），实例化对应的 `RobotEnvSpec`（如 `Turtlebot3Spec`）。
- 通过 ROS 话题 / 服务与 Gazebo 交互：
    - 发布 `/cmd_vel`；
    - 订阅 odom / scan / contact；
    - 调用 `/gazebo/reset_world`、`/gazebo/pause_physics` 等服务。
- 实现经典的 env 接口：
    - `reset() -> np.ndarray`：
        - 重置世界和机器人位姿。
        - 等待传感器数据就绪。
        - 调 `spec.build_observation(...)` 得到初始观测 `obs_np`。
    - `step(action_np) -> (obs_np, reward, done, info)`：
        - 用 `spec.command_from_action` 将动作映射到 `Twist` 并发布。
        - 推进仿真若干物理时间步，等待新的传感器数据；
        - 调 `spec.build_observation` 生成新的观测；
        - 调 `spec.compute_reward()` 计算 reward；
        - 调 `spec.is_done()` 判断是否终止；
        - 返回 `(obs_np, reward, done, info)`。

这一层**不关心奖励的细节**，只负责“调用 spec 完成观测和 reward 的计算”。

### 3. VecEnv 适配层：`GazeboSingleVecEnv`

文件：`scripts/gazebo_rl_gym/envs/gazebo_vec_env.py`

职责：把 `GazeboSingleRobotEnv` 包装成 `rsl_rl.env.VecEnv`，接口如下：

- `reset() -> TensorDict`
    - 调用单车 env 的 `reset()` 拿到 `obs_np`（shape `[obs_dim]`）。
    - 转成 `obs_tensor = torch.as_tensor(obs_np).unsqueeze(0)`，得到 `[1, obs_dim]`。
    - 用 `TensorDict` 打包成：
        ```python
        TensorDict({
                "policy": obs_tensor,
                "critic": obs_tensor,
        }, batch_size=[1])
        ```

- `step(actions) -> (obs_td, rewards, dones, extras)`
    - `actions` 来自 PPO，为 `[1, num_actions]`。
    - 转为 numpy 后传给单车 env 的 `step()`。
    - 将返回的 `next_obs_np` 转为 TensorDict（同上）。
    - `reward` / `done` 转成形状为 `[num_envs] = [1]` 的 tensor：
        ```python
        rewards = torch.as_tensor([reward], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([done], dtype=torch.bool, device=self.device)
        ```
    - 维护一个 `episode_length_buf` 计数步数，如果你不希望 timeout，可以保持 `time_outs=False` 或让 episode 完全由 `spec.is_done` 决定。
    - 返回 `(obs_td, rewards, dones, extras)`。

### 4. 算法 / Runner 层：`OnPolicyRunner(PPO)`

文件：`rsl_rl/rsl_rl/runners/on_policy_runner.py`（外部库）

初始化：

- 调 `env.get_observations()` 获取一帧样本观测 `obs_td`。
- 根据 `obs_groups` 配置解析出给 actor / critic 使用的不同观测分组。
- 基于 `obs_td` 的 shape 构建 ActorCritic 网络，其输入维度自动等于你的 `observation_dim`。
- 构建 PPO 算法对象 `alg`，并初始化 RolloutStorage。

训练循环 `learn()`：

- 每个 iteration：
    - 采样 `num_steps_per_env` 步：
        - 用当前 `obs` 调 `self.alg.act(obs)` 得到 `actions`；
        - 调 `env.step(actions)` 获取 `next_obs`, `rewards`, `dones`, `extras`；
        - 调 `self.alg.process_env_step(...)` 把这一步存入 RolloutStorage；
        - 记录 `cur_reward_sum += rewards` 和 `cur_episode_length += 1`，用于日志。
    - 结束后调 `self.alg.compute_returns(obs)` 计算回报；
    - 调 `self.alg.update()` 进行 PPO 更新，更新策略网络和价值网络。

到此，观测从 `Turtlebot3Spec` 流到 MLP 输入；奖励从 `compute_reward` 流到 PPO 的 loss 计算，全流程闭环。

## 目标点导航：在机器人坐标系中加入目标角度和距离奖励

导航任务通常需要两个关键观测：

- 机器人到目标点的**距离** $d$；
- 目标在机器人坐标系中的**方位角** $\theta$（例如前方为 0，左正右负）。

并配套几个奖励：

- 距离缩短奖励（靠近目标）；
- 到达终点时的一次性大正奖励；
- 每一步时间成本（负奖励），防止原地空等。

### 1. 在 YAML 中配置目标点和奖励系数

示例：修改 `config/envs/warehouse.yaml` 中 `robots` 段落：

```yaml
robots:
    - name: turtle1
        preset: turtlebot3_waffle
        pose: {x: 0.0, y: 0.0, z: 0.1, yaw: 0.0}
        task:
            target_pos: [2.0, 1.0]       # 目标点 (x, y)
            success_radius: 0.3          # 终点判定半径
        reward:
            distance_scale: 0.0          # 如不再用原始位移奖励，可设 0
            smoothness_scale: 0.01       # 动作平滑惩罚
            collision_penalty: 10.0      # 碰撞惩罚
            min_range_penalty: 2.0
            min_range_threshold: 0.3
            goal_reward: 20.0            # 到达终点一次性奖励
            dist_reward_scale: 1.0       # 距离缩短奖励系数
            step_cost: 0.01              # 每步时间成本（负奖励）
```

这里**不再依赖 `max_episode_length` 做 timeout**，而是用 reward 里的 `step_cost` 让每一步都有代价，鼓励尽快到达目标。

### 2. 在 `Turtlebot3Spec` 中加入目标距离和角度观测

在 `__init__` 中读取任务配置，并初始化状态：

```python
class Turtlebot3Spec(RobotEnvSpec):
        POSE_KEYS = ("x", "y", "yaw", "vx", "vy", "wz")
        SCAN_DIM = 360

        def __init__(self, name: str, cfg):
                super().__init__(name, cfg)
                self._current_height = 0.0
                self._last_scan_min_value: Optional[float] = None

                task_cfg = getattr(cfg, "task", None) or {}
                target = task_cfg.get("target_pos", [1.0, 0.0])
                self.target_x = float(target[0])
                self.target_y = float(target[1])
                self.success_radius = float(task_cfg.get("success_radius", 0.3))

                self._prev_dist: float | None = None
```

修改 `observation_dim`，在原有基础上加 2 维（距离 + 目标角度）：

```python
        @property
        def observation_dim(self) -> int:
                base_dim = len(self.POSE_KEYS) + self.SCAN_DIM
                return base_dim + 2  # + [dist, target_angle]
```

在 `build_observation` 末尾拼接距离和角度：

```python
        def build_observation(self, pose_msg: Odometry, scan_msg: Optional[LaserScan]) -> np.ndarray:
                translation = pose_msg.pose.pose.position
                rotation = pose_msg.pose.pose.orientation
                roll, pitch, yaw = euler_from_quaternion([
                        rotation.x, rotation.y, rotation.z, rotation.w
                ])

                mapping = { ... }  # 保持你现有实现
                components = [mapping[key] for key in self.POSE_KEYS]
                pose_vec = np.array(components, dtype=np.float32)

                # 激光同你现有代码...
                # scan = ...

                self._last_scan_min_value = float(np.min(scan))
                self._current_height = float(translation.z)

                # 目标在世界坐标中的偏差
                dx = self.target_x - translation.x
                dy = self.target_y - translation.y
                dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)

                # 转到机器人坐标系：目标方向角
                target_angle = np.arctan2(dy, dx) - yaw
                target_angle = (target_angle + np.pi) % (2 * np.pi) - np.pi
                target_angle = target_angle.astype(np.float32)

                parts = [pose_vec, scan, np.array([dist, target_angle], dtype=np.float32)]
                return np.concatenate([p for p in parts if p.size])
```

### 3. 在 reward 中加入：距离缩短 + 终点奖励 + 每步成本

在 `compute_reward` 中扩展：

```python
        def compute_reward(self) -> float:
                reward = 0.0

                # 原有：位移、平滑、碰撞、最小距离等
                # ... 保留你现在的实现

                # --- 距离缩短奖励 ---
                curr = self.current_observation
                pose_dim = len(self.POSE_KEYS)
                curr_pose = curr[:pose_dim]
                x, y = float(curr_pose[0]), float(curr_pose[1])

                dx = self.target_x - x
                dy = self.target_y - y
                dist = float(np.sqrt(dx * dx + dy * dy))

                if self._prev_dist is not None:
                        delta = self._prev_dist - dist  # 距离减小为正
                        dist_scale = getattr(self.cfg.reward, "dist_reward_scale", 1.0)
                        reward += dist_scale * delta
                self._prev_dist = dist

                # --- 到终点奖励（一次性） ---
                if dist < self.success_radius:
                        goal_reward = getattr(self.cfg.reward, "goal_reward", 20.0)
                        reward += goal_reward

                # --- 每步时间成本 ---
                step_cost = getattr(self.cfg.reward, "step_cost", 0.0)
                if step_cost > 0.0:
                        reward -= step_cost

                return reward
```

### 4. 终止条件：仅由碰撞 / 离开高度 / 到达终点控制（无 timeout）

在 `is_done` 中加入终点判定：

```python
        def is_done(self) -> bool:
                # 碰撞终止
                if getattr(self.cfg.termination, "collision", False) and self.collision_active:
                        return True

                # 高度终止
                min_height = getattr(self.cfg.termination, "min_height", None)
                if min_height is not None and self._current_height < float(min_height):
                        return True

                # 终点终止
                if self._prev_dist is not None and self._prev_dist < self.success_radius:
                        return True

                return False
```

同时，在 `GazeboSingleVecEnv` 中，你可以：

- 不再使用 `max_episode_length` 作为 timeout 判定；
- `time_outs` 始终设为 `False` 或保持为全零张量，让 episode 完全由 `is_done` 控制。

这样，**没有 timeout**，但通过 `step_cost` 让每一步都有负代价，防止原地空等；当车到达目标点时终止回合并给予一次性的 `goal_reward` 激励。

## 训练 YAML 关键字段说明

以 `config/train/warehouse_ppo_turtlebot3.yaml` 为例：

- `env_cfg`：环境配置文件路径（world + 机器人 + 任务参数）。
- `num_steps_per_env`：每个 iteration、每个 env 采样的步数（rollout 长度）。
- `num_iterations`：总训练迭代次数，总步数约为 `num_steps_per_env * num_iterations`（单 env）。
- `save_interval`：每隔多少 iteration 保存一次模型。
- `logger`：日志方式（`tensorboard` / `wandb` / `neptune`）。
- `algorithm`：PPO 的超参数（折扣因子 `gamma`、学习率、clip 参数、loss 系数等）。
- `policy`：ActorCritic 网络结构（隐藏层大小、激活函数、是否做观测归一化等）。
- `obs_groups`：把 `TensorDict` 中各个 key 的观测分配给 actor / critic 使用，目前采用简单配置：
    - `actor: ["policy"]`
    - `critic: ["critic"]`

episode 的长度现在主要由：

- `Turtlebot3Spec.is_done()` 中的碰撞 / 高度 / 到达终点判断；
- reward 中的 `step_cost` 施加时间压力，而不是 `max_episode_length` 超时。

建议在调参时从以下几个方向逐步调整：

- `goal_reward`：越大，越强烈地鼓励到达目标；
- `dist_reward_scale`：决定靠近目标的细腻程度；
- `step_cost`：越大，越鼓励尽快到达（每多走一步多付代价）；
- `collision_penalty` / `min_range_penalty`：控制安全性与试探性之间的平衡。
