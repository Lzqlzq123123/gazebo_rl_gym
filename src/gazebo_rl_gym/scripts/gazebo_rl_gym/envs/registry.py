from __future__ import annotations

from typing import Dict, Tuple, Type

from .base.robot_config import RobotCfg
from .base.robot_spec import RobotEnvSpec


class RobotRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Tuple[Type[RobotEnvSpec], Type[RobotCfg]]] = {}

    def register(self, name: str, spec_cls: Type[RobotEnvSpec], cfg_cls: Type[RobotCfg]) -> None:
        if name in self._registry:
            raise ValueError(f"Robot preset '{name}' already registered")
        self._registry[name] = (spec_cls, cfg_cls)

    def get(self, name: str) -> Tuple[Type[RobotEnvSpec], Type[RobotCfg]]:
        if name not in self._registry:
            raise KeyError(f"Robot preset '{name}' has not been registered")
        return self._registry[name]

    def create(self, preset_name: str, robot_name: str, overrides: dict | None = None) -> RobotEnvSpec:
        spec_cls, cfg_cls = self.get(preset_name)
        cfg = cfg_cls()
        print(f"Debug - Initial cfg for {preset_name}: {cfg}")
        if overrides:
            from .base.config_utils import update_config_from_dict
            print(f"Debug - Applying overrides to {preset_name}: {overrides}")
            update_config_from_dict(cfg, overrides)
            print(f"Debug - Final cfg after overrides for {preset_name}: {cfg}")
        return spec_cls(robot_name, cfg)


robot_registry = RobotRegistry()
