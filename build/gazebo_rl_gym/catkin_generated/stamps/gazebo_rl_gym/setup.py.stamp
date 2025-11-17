from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[
        'gazebo_rl_gym',
        'gazebo_rl_gym.algo',
        'gazebo_rl_gym.algo.ppo',
        'gazebo_rl_gym.envs',
        'gazebo_rl_gym.utils',
    ],
    package_dir={'': 'scripts'}
)

setup(**d)
