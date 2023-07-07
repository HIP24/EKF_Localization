# pamuk

## Dependencies

This project depends on the following ROS packages:

- gazebo_ros
- xacro
- turtlebot3_description
- turtlebot3_bringup
- map_server
- tf
- rviz
- move_base



## Installation

To install these dependencies, run the following commands:

```
$sudo apt-get install ros-noetic-gazebo-ros  
$sudo apt-get install ros-noetic-xacro  
$sudo apt-get install ros-noetic-turtlebot3-description  
$sudo apt-get install ros-noetic-turtlebot3-bringup  
$sudo apt-get install ros-noetic-map-server  
$sudo apt-get install ros-noetic-tf  
$sudo apt-get install ros-noetic-rviz  
$sudo apt-get install ros-noetic-move-base  
```


## Set turtlebot model and gazebo model path (to find the map) in bashrc

Open bashrc with:

```
$sudo nano ~/.bashrc  
```

Add following entries:   
`export TURTLEBOT3_MODEL=waffle`  
`export GAZEBO_MODEL_PATH=/path/to/package/pamuk`  

Finally, either **open a new terminal** or `source ~/.bashrc`

## Build and launch project

Build it using `catkin_make` or `catkin build` from root.

Launch this project with `roslaunch pamuk pamuk.launch`
