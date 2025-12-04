
from .registry import robot_registry
from .turtlebot3.turtlebot3_config import Turtlebot3Cfg
from .turtlebot3.turtlebot3_env import Turtlebot3Spec
from .nanocar.nanocar_config import NanocarCfg
from .nanocar.nanocar_env import NanocarSpec
from .forklift.forklift_config import ForkliftCfg
from .forklift.forklift_env import ForkliftSpec

robot_registry.register("turtlebot3_waffle", Turtlebot3Spec, Turtlebot3Cfg)
robot_registry.register("nanocar", NanocarSpec, NanocarCfg)
robot_registry.register("forklift", ForkliftSpec, ForkliftCfg)

__all__ = [
	"robot_registry",
	"Turtlebot3Cfg",
	"Turtlebot3Spec",
	"NanocarCfg",
	"NanocarSpec",
	"ForkliftCfg",
	"ForkliftSpec",
]
