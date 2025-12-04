cmake_minimum_required(VERSION 3.0.2)
project(gazebo_rl_gym)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_python_setup()

catkin_package()

install(DIRECTORY launch config urdf meshes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS scripts/train.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
