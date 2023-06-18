cmake_minimum_required(VERSION 2.8.3)
project(my_cool_project)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  sensor_msgs
  std_msgs
  roscpp
  rosconsole
  actionlib
  tf2
  tf2_ros
  tf2_geometry_msgs
)

find_package(Eigen3 REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES my_cool_project
  CATKIN_DEPENDS roscpp rosconsole actionlib tf2 tf2_ros nav_msgs sensor_msgs std_msgs
  DEPENDS EIGEN3
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(Move_Turtlebot src/Move_Turtlebot.cpp)
target_link_libraries(Move_Turtlebot ${catkin_LIBRARIES} pthread)

add_executable(EKF_Loc src/EKF_Loc.cpp)
target_link_libraries(EKF_Loc ${catkin_LIBRARIES} pthread)