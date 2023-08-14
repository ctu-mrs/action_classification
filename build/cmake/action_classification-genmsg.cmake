# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "action_classification: 3 messages, 0 services")

set(MSG_I_FLAGS "-Iaction_classification:/home/akash/swarm_ws/src/action_classification/msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg;-Inav_msgs:/opt/ros/noetic/share/nav_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/mpc_tracker;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/landoff_tracker;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/mrs_hw_modules;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/odometry;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/ouster;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/uav_managers;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/uav_status;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/se3_controller;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/trajectory_generation;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/diagnostics;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/gnss;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/mrs_serial;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/mrs_gripper;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/parachute;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/profiler;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/bumper;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/speed_tracker;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/uvdar;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/sxd;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/general;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/simulation;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/pathfinder;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/stamped_msgs;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/mrs_pcl_tools;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/mrs_llcp;-Imrs_msgs:/home/akash/mrs_workspace/src/uav_core/ros_packages/mrs_msgs/msg/radar;-Iactionlib_msgs:/opt/ros/noetic/share/actionlib_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(action_classification_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/landmark.msg" NAME_WE)
add_custom_target(_action_classification_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "action_classification" "/home/akash/swarm_ws/src/action_classification/msg/landmark.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/IntStamped.msg" NAME_WE)
add_custom_target(_action_classification_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "action_classification" "/home/akash/swarm_ws/src/action_classification/msg/IntStamped.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/landmark3D.msg" NAME_WE)
add_custom_target(_action_classification_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "action_classification" "/home/akash/swarm_ws/src/action_classification/msg/landmark3D.msg" "std_msgs/Header"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(action_classification
  "/home/akash/swarm_ws/src/action_classification/msg/landmark.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/action_classification
)
_generate_msg_cpp(action_classification
  "/home/akash/swarm_ws/src/action_classification/msg/IntStamped.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/action_classification
)
_generate_msg_cpp(action_classification
  "/home/akash/swarm_ws/src/action_classification/msg/landmark3D.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/action_classification
)

### Generating Services

### Generating Module File
_generate_module_cpp(action_classification
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/action_classification
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(action_classification_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(action_classification_generate_messages action_classification_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/landmark.msg" NAME_WE)
add_dependencies(action_classification_generate_messages_cpp _action_classification_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/IntStamped.msg" NAME_WE)
add_dependencies(action_classification_generate_messages_cpp _action_classification_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/landmark3D.msg" NAME_WE)
add_dependencies(action_classification_generate_messages_cpp _action_classification_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(action_classification_gencpp)
add_dependencies(action_classification_gencpp action_classification_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS action_classification_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(action_classification
  "/home/akash/swarm_ws/src/action_classification/msg/landmark.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/action_classification
)
_generate_msg_eus(action_classification
  "/home/akash/swarm_ws/src/action_classification/msg/IntStamped.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/action_classification
)
_generate_msg_eus(action_classification
  "/home/akash/swarm_ws/src/action_classification/msg/landmark3D.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/action_classification
)

### Generating Services

### Generating Module File
_generate_module_eus(action_classification
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/action_classification
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(action_classification_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(action_classification_generate_messages action_classification_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/landmark.msg" NAME_WE)
add_dependencies(action_classification_generate_messages_eus _action_classification_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/IntStamped.msg" NAME_WE)
add_dependencies(action_classification_generate_messages_eus _action_classification_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/landmark3D.msg" NAME_WE)
add_dependencies(action_classification_generate_messages_eus _action_classification_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(action_classification_geneus)
add_dependencies(action_classification_geneus action_classification_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS action_classification_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(action_classification
  "/home/akash/swarm_ws/src/action_classification/msg/landmark.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/action_classification
)
_generate_msg_lisp(action_classification
  "/home/akash/swarm_ws/src/action_classification/msg/IntStamped.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/action_classification
)
_generate_msg_lisp(action_classification
  "/home/akash/swarm_ws/src/action_classification/msg/landmark3D.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/action_classification
)

### Generating Services

### Generating Module File
_generate_module_lisp(action_classification
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/action_classification
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(action_classification_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(action_classification_generate_messages action_classification_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/landmark.msg" NAME_WE)
add_dependencies(action_classification_generate_messages_lisp _action_classification_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/IntStamped.msg" NAME_WE)
add_dependencies(action_classification_generate_messages_lisp _action_classification_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/landmark3D.msg" NAME_WE)
add_dependencies(action_classification_generate_messages_lisp _action_classification_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(action_classification_genlisp)
add_dependencies(action_classification_genlisp action_classification_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS action_classification_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(action_classification
  "/home/akash/swarm_ws/src/action_classification/msg/landmark.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/action_classification
)
_generate_msg_nodejs(action_classification
  "/home/akash/swarm_ws/src/action_classification/msg/IntStamped.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/action_classification
)
_generate_msg_nodejs(action_classification
  "/home/akash/swarm_ws/src/action_classification/msg/landmark3D.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/action_classification
)

### Generating Services

### Generating Module File
_generate_module_nodejs(action_classification
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/action_classification
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(action_classification_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(action_classification_generate_messages action_classification_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/landmark.msg" NAME_WE)
add_dependencies(action_classification_generate_messages_nodejs _action_classification_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/IntStamped.msg" NAME_WE)
add_dependencies(action_classification_generate_messages_nodejs _action_classification_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/landmark3D.msg" NAME_WE)
add_dependencies(action_classification_generate_messages_nodejs _action_classification_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(action_classification_gennodejs)
add_dependencies(action_classification_gennodejs action_classification_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS action_classification_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(action_classification
  "/home/akash/swarm_ws/src/action_classification/msg/landmark.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/action_classification
)
_generate_msg_py(action_classification
  "/home/akash/swarm_ws/src/action_classification/msg/IntStamped.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/action_classification
)
_generate_msg_py(action_classification
  "/home/akash/swarm_ws/src/action_classification/msg/landmark3D.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/action_classification
)

### Generating Services

### Generating Module File
_generate_module_py(action_classification
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/action_classification
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(action_classification_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(action_classification_generate_messages action_classification_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/landmark.msg" NAME_WE)
add_dependencies(action_classification_generate_messages_py _action_classification_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/IntStamped.msg" NAME_WE)
add_dependencies(action_classification_generate_messages_py _action_classification_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/akash/swarm_ws/src/action_classification/msg/landmark3D.msg" NAME_WE)
add_dependencies(action_classification_generate_messages_py _action_classification_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(action_classification_genpy)
add_dependencies(action_classification_genpy action_classification_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS action_classification_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/action_classification)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/action_classification
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(action_classification_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()
if(TARGET nav_msgs_generate_messages_cpp)
  add_dependencies(action_classification_generate_messages_cpp nav_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(action_classification_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(action_classification_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET mrs_msgs_generate_messages_cpp)
  add_dependencies(action_classification_generate_messages_cpp mrs_msgs_generate_messages_cpp)
endif()
if(TARGET std_srvs_generate_messages_cpp)
  add_dependencies(action_classification_generate_messages_cpp std_srvs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/action_classification)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/action_classification
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(action_classification_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()
if(TARGET nav_msgs_generate_messages_eus)
  add_dependencies(action_classification_generate_messages_eus nav_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(action_classification_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(action_classification_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET mrs_msgs_generate_messages_eus)
  add_dependencies(action_classification_generate_messages_eus mrs_msgs_generate_messages_eus)
endif()
if(TARGET std_srvs_generate_messages_eus)
  add_dependencies(action_classification_generate_messages_eus std_srvs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/action_classification)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/action_classification
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(action_classification_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()
if(TARGET nav_msgs_generate_messages_lisp)
  add_dependencies(action_classification_generate_messages_lisp nav_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(action_classification_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(action_classification_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET mrs_msgs_generate_messages_lisp)
  add_dependencies(action_classification_generate_messages_lisp mrs_msgs_generate_messages_lisp)
endif()
if(TARGET std_srvs_generate_messages_lisp)
  add_dependencies(action_classification_generate_messages_lisp std_srvs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/action_classification)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/action_classification
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(action_classification_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()
if(TARGET nav_msgs_generate_messages_nodejs)
  add_dependencies(action_classification_generate_messages_nodejs nav_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(action_classification_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(action_classification_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET mrs_msgs_generate_messages_nodejs)
  add_dependencies(action_classification_generate_messages_nodejs mrs_msgs_generate_messages_nodejs)
endif()
if(TARGET std_srvs_generate_messages_nodejs)
  add_dependencies(action_classification_generate_messages_nodejs std_srvs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/action_classification)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/action_classification\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/action_classification
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(action_classification_generate_messages_py geometry_msgs_generate_messages_py)
endif()
if(TARGET nav_msgs_generate_messages_py)
  add_dependencies(action_classification_generate_messages_py nav_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(action_classification_generate_messages_py sensor_msgs_generate_messages_py)
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(action_classification_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET mrs_msgs_generate_messages_py)
  add_dependencies(action_classification_generate_messages_py mrs_msgs_generate_messages_py)
endif()
if(TARGET std_srvs_generate_messages_py)
  add_dependencies(action_classification_generate_messages_py std_srvs_generate_messages_py)
endif()
