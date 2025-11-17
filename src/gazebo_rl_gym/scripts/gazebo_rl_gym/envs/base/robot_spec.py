from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ContactsState

from .robot_config import RobotCfg


class RobotEnvSpec(ABC):
    cfg_cls = RobotCfg

    def __init__(self, name: str, cfg: RobotCfg):
        self.name = name
        self.cfg = cfg
        self._current_obs: Optional[np.ndarray] = None
        self._prev_obs: Optional[np.ndarray] = None
        self._current_action: Optional[np.ndarray] = None
        self._prev_action: Optional[np.ndarray] = None
        self._collision_active: bool = False
        self._last_contact_msg: Optional[ContactsState] = None

    # --- convenience -------------------------------------------------
    @property
    @abstractmethod
    def observation_dim(self) -> int:
        """Return the dimension of the observation vector produced by this spec."""

    @property
    def uses_scan(self) -> bool:
        topic_template = getattr(self.cfg.topics, "scan", "")
        return bool(topic_template)

    @property
    def uses_contact(self) -> bool:
        topic_template = getattr(self.cfg.topics, "contact", "")
        return bool(topic_template)

    @property
    def action_dim(self) -> int:
        return len(self.cfg.action_space.low)

    @property
    def cmd_topic(self) -> str:
        return self._ensure_abs_topic(self.cfg.topics.cmd.format(name=self.name))

    @property
    def pose_topic(self) -> str:
        return self._ensure_abs_topic(self.cfg.topics.pose.format(name=self.name))

    @property
    def scan_topic(self) -> str:
        topic_template = getattr(self.cfg.topics, "scan", "")
        if not topic_template:
            return ""
        return self._ensure_abs_topic(topic_template.format(name=self.name))

    @property
    def contact_topic(self) -> str:
        topic_template = getattr(self.cfg.topics, "contact", "")
        if not topic_template:
            return ""
        return self._ensure_abs_topic(topic_template.format(name=self.name))

    @property
    def action_low(self) -> np.ndarray:
        return np.asarray(self.cfg.action_space.low, dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.asarray(self.cfg.action_space.high, dtype=np.float32)

    def reset(self) -> None:
        self._prev_obs = None
        self._current_obs = None
        self._prev_action = None
        self._current_action = None
        self._collision_active = False
        self._last_contact_msg = None

    # --- sensing -----------------------------------------------------
    def process_sensor_data(self, pose_msg: Odometry, scan_msg: Optional[LaserScan]) -> np.ndarray:
        obs = self.build_observation(pose_msg, scan_msg)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        self._prev_obs = self._current_obs
        self._current_obs = obs
        return obs

    @abstractmethod
    def build_observation(self, pose_msg: Odometry, scan_msg: Optional[LaserScan]) -> np.ndarray:
        """Convert raw sensor messages into an observation vector."""

    # --- reward & termination ---------------------------------------
    @abstractmethod
    def compute_reward(self) -> float:
        """Return the reward for the most recent transition."""

    @abstractmethod
    def is_done(self) -> bool:
        """Return True if the current episode should terminate."""

    # --- actions -----------------------------------------------------
    def apply_action(self, action: np.ndarray) -> Twist:
        """Convert the agent action into a command and record action history."""
        action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_vec.size == 0:
            raise ValueError(f"Action for robot '{self.name}' must not be empty")

        if self._current_action is not None:
            self._prev_action = self._current_action.copy()
        else:
            self._prev_action = None

        self._current_action = action_vec.copy()
        return self.command_from_action(action_vec)

    @abstractmethod
    def command_from_action(self, action: np.ndarray) -> Twist:
        """Map the processed action vector to a ROS Twist command."""

    # --- helpers -----------------------------------------------------
    @property
    def current_observation(self) -> Optional[np.ndarray]:
        return None if self._current_obs is None else self._current_obs.copy()

    @property
    def previous_observation(self) -> Optional[np.ndarray]:
        return None if self._prev_obs is None else self._prev_obs.copy()

    @property
    def current_action(self) -> Optional[np.ndarray]:
        return None if self._current_action is None else self._current_action.copy()

    @property
    def previous_action(self) -> Optional[np.ndarray]:
        return None if self._prev_action is None else self._prev_action.copy()

    @property
    def collision_active(self) -> bool:
        return self._collision_active

    @property
    def last_contact_msg(self) -> Optional[ContactsState]:
        return self._last_contact_msg

    def update_contact_state(self, contacts: ContactsState | None) -> None:
        if contacts is None:
            self._collision_active = False
            self._last_contact_msg = None
            return

        self._collision_active = bool(contacts.states)
        self._last_contact_msg = contacts

    @staticmethod
    def _ensure_abs_topic(topic: str) -> str:
        if not topic:
            return topic
        return topic if topic.startswith("/") else f"/{topic}"


