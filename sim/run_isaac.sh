#!/usr/bin/env bash
# Run a Python script with Isaac Sim's interpreter and its bundled ROS 2 Humble libraries.
# Usage: sim/run_isaac.sh sim/isaac_portrait_sim.py [args]
# Override the interpreter with ISAAC_PYTHON=/path/to/python.
set -eo pipefail
ISAAC_PYTHON="${ISAAC_PYTHON:-$HOME/miniconda3/envs/isaaclab3/bin/python}"
SITE=$("$ISAAC_PYTHON" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
ROS_INTERNAL="$SITE/isaacsim/exts/isaacsim.ros2.core/humble"
# Drop the system ROS (Python 3.10) environment; Isaac Sim uses its own Python 3.12 build.
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH="$ROS_INTERNAL/lib"
export PYTHONPATH="$ROS_INTERNAL/rclpy"
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export OMNI_KIT_ACCEPT_EULA=YES
# The lab machine needs user-space NVIDIA GL/Vulkan libraries matching the kernel driver
# (otherwise Vulkan only sees the Intel iGPU and RTX rendering fails). Override with ISAAC_GL_ENV.
GL_ENV="${ISAAC_GL_ENV:-$HOME/nvidia-gl-580-196/env.sh}"
if [ -f "$GL_ENV" ]; then source "$GL_ENV"; fi
# Over ssh, use the local desktop X server (:1 on the lab machine) instead of an X-forwarded
# display: the NVIDIA Vulkan driver needs it here. Override with ISAAC_DISPLAY.
case "${DISPLAY:-}" in
  ""|localhost:*)
    export DISPLAY="${ISAAC_DISPLAY:-:1}"
    if [ -z "${XAUTHORITY:-}" ] || [ ! -f "${XAUTHORITY}" ]; then
      [ -f "/run/user/$(id -u)/gdm/Xauthority" ] && export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
    fi ;;
esac
exec "$ISAAC_PYTHON" "$@"
