#!/usr/bin/env bash
# Stop everything started by portrait.launch.py (sim or real) and the post-processing service,
# then report anything still running. Safe to run any time; it never sends motion to the robot
# (the arm node stops the robot's motion queue as it shuts down).
pkill -INT -f "[r]os2 launch portrait_bringup" || true
for i in $(seq 1 30); do pgrep -f "[p]ortrait_arm/arm_node" >/dev/null || break; sleep 0.5; done
pkill -f "[i]saac_portrait_sim" || true
pkill -f "[p]ortrait_arm/arm_node|[p]ortrait_session/session_node|[p]ortrait_perception/perception_node|[p]ortrait_voice/voice_node|[p]ortrait_camera/camera_node|[r]obot_state_publisher" || true
pkill -INT -f "[p]ortrait_post.watch" || true
pkill -INT -f "[t]ts_server.py" || true
sleep 2
left=$(pgrep -fa "[r]os2 launch portrait_bringup|[i]saac_portrait_sim|[p]ortrait_(arm|session|perception|voice|camera)/|[r]obot_state_publisher|[p]ortrait_post.watch|[t]ts_server.py")
if [ -n "$left" ]; then
  echo "Still running (kill these by PID):"; echo "$left"; exit 1
fi
echo "All portrait processes stopped."
