from __future__ import annotations


#from sklearn.tree import ExtraTreeClassifier

"""VecEnv adapter for a single Gazebo robot, compatible with rsl_rl.

This wraps GazeboSingleRobotEnv and exposes the VecEnv API expected by
rsl_rl.runners.OnPolicyRunner. It does not depend on gym.
"""

from typing import Dict, Any

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from .gazebo_single_env import GazeboSingleRobotEnv


class GazeboSingleVecEnv(VecEnv):
    """Single-environment VecEnv wrapper around GazeboSingleRobotEnv."""

    def __init__(self, env_cfg_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.num_envs = 1

        # Underlying Gazebo environment
        self.env = GazeboSingleRobotEnv(env_cfg_path)

        # Probe one reset to infer obs/action dimensions once implemented
        # For now we assume the GazeboSingleRobotEnv will set obs_dim/action_dim
        self.obs_dim = getattr(self.env, "obs_dim", None)
        self.num_actions = getattr(self.env, "action_dim", None)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Minimal env config for logging
        self.cfg: Dict[str, Any] = {"env_name": "GazeboSingleRobot", "env_cfg_path": env_cfg_path}

    # ------------------------------------------------------------------
    # VecEnv API
    # ------------------------------------------------------------------
    def reset(self) -> TensorDict:
        obs_np = self.env.reset()
        self.episode_length_buf.zero_()
        return self._obs_to_tensordict(obs_np)

    def step(self, actions: torch.Tensor):
        """Step the underlying Gazebo env.

        actions: (1, num_actions)
        Returns: obs_td, rewards, dones, extras
        """
        act_np = actions[0].detach().cpu().numpy()
        next_obs_np, reward, done, info = self.env.step(act_np)
        reward = float(reward)
        done = bool(done)
        self.episode_length_buf += 1


        obs_td = self._obs_to_tensordict(next_obs_np)
        rewards = torch.as_tensor([reward], dtype=torch.float32, device=self.device)  # shape [1]
        dones = torch.as_tensor([done], dtype=torch.bool, device=self.device)         # shape [1]

        extras: Dict[str, Any] = {}
        if "episode" in info:
            extras["episode"] = info["episode"]
            extras["episode"]["length"] = int(self.episode_length_buf[0].item())
        elif "log" in info:
            extras["log"] = info["log"]

        return obs_td, rewards, dones, extras

    def get_observations(self) -> TensorDict:
        # For now we simply call reset(); later this can reuse the latest obs
        obs_np = self.env.reset()
        return self._obs_to_tensordict(obs_np)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _obs_to_tensordict(self, obs_np) -> TensorDict:
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        return TensorDict(
            {
                "policy": obs_tensor,
                "critic": obs_tensor,
            },
            batch_size=[self.num_envs],
            device=self.device,
        )
