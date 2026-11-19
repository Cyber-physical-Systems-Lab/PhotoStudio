cmake_minimum_required(VERSION 3.8)
project(portrait_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FaceObservation.msg"
  "msg/ArmStatus.msg"
  "srv/MoveCamera.srv"
  "srv/MoveNamed.srv"
  "srv/Say.srv"
  "srv/StartSession.srv"
  "srv/SaveStill.srv"
  DEPENDENCIES std_msgs geometry_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
