cmake_minimum_required(VERSION 2.8.3)
project(costar_gazebo_plugins)

find_package(catkin REQUIRED COMPONENTS 
  message_generation 
  gazebo_msgs 
  roscpp 
  rospy 
  std_srvs 
  geometry_msgs 
  sensor_msgs 
  urdf 
  tf 
  tf2_ros 
  dynamic_reconfigure 
  trajectory_msgs 
  pcl_conversions
  image_transport 
  rosconsole
  diagnostic_updater
  moveit_msgs
)

#include (FindPkgConfig)
#if (PKG_CONFIG_FOUND)
#  pkg_check_modules(XML libxml-2.0)
#  pkg_check_modules(OGRE OGRE)
#  pkg_check_modules(OGRE-Terrain OGRE-Terrain)
#else()
#  message(FATAL_ERROR "pkg-config is required; please install it")
#endif()

# Depend on system install of Gazebo and SDFormat
find_package(gazebo REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)

execute_process(COMMAND
  pkg-config --variable=plugindir OGRE
  OUTPUT_VARIABLE OGRE_PLUGIN_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

include_directories(include
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
  ${OGRE_INCLUDE_DIRS}
  ${OGRE-Terrain_INCLUDE_DIRS}
)

link_directories(
  ${GAZEBO_LIBRARY_DIRS}
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES 
  
  CATKIN_DEPENDS 
  message_generation 
  gazebo_msgs 
  roscpp 
  rospy 
  nodelet 
  angles 
  std_srvs 
  geometry_msgs 
  sensor_msgs 
  nav_msgs 
  urdf 
  tf 
  tf2_ros
  rosgraph_msgs 
  trajectory_msgs 
  pcl_conversions
  image_transport 
  rosconsole
  moveit_msgs
  DEPENDS 
    gazebo 
    SDF
  )

add_definitions(-fPIC) # what is this for?

## Plugins
add_library(gazebo_ros_moveit_planning_scene src/gazebo_ros_moveit_planning_scene.cpp)
target_link_libraries(gazebo_ros_moveit_planning_scene ${GAZEBO_LIBRARIES} ${catkin_LIBRARIES})

install(TARGETS 
  gazebo_ros_moveit_planning_scene
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  )

install(DIRECTORY include/
   DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
   FILES_MATCHING PATTERN "*.h"
)

