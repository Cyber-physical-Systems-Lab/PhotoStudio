# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Implemented 2026-11-14 and run end to end in Isaac Sim on the lab machine (see README.md for the
layout, safety list and commands, docs/monday_runbook.md for the first hardware test).

## Commands

- Work happens on the lab machine (`ssh lab`, checkout at `~/Nova5Photo`); this folder is synced there
  with `rsync -a --delete --exclude .venv --exclude .venv-tts --exclude voices --exclude build --exclude install --exclude log --exclude portraits --exclude layout --exclude .git ./ lab:Nova5Photo/`.
- Build: `source /opt/ros/humble/setup.bash && colcon build --symlink-install`
- Sim run (detached, logs in /tmp/n5run/launch.log, video in /tmp/n5run/sim_run.mp4): `scripts/lab_run_sim.sh /tmp/n5run`; stop with `scripts/lab_stop.sh`
- Real: `ros2 launch portrait_bringup portrait.launch.py mode:=real robot_ip:=192.168.0.37 camera:=/dev/video0`
- Tests: `cd src/<pkg> && PYTHONPATH=. python3 -m pytest -q test` (arm, session, perception, voice, post)

## Gotchas found while building

- Isaac Sim 6.1 lives in the `isaaclab3` conda env; always launch through `sim/run_isaac.sh` (bundled
  ROS 2 Humble rclpy, user-space NVIDIA GL from `~/nvidia-gl-580-196/env.sh`, `DISPLAY=:1`).
- `omni.usd...new_stage()` crashes OmniGraph in this build; use the startup stage.
- DLSS/NGX and the OptiX denoiser fail on the lab machine; use the PathTracing renderer (8 spp).
- The robot in sim is kinematic (link transforms from our FK); the importer's physics variant is off.
- Large images must use reliable QoS; best-effort drops almost every multi-MB frame.
- The A4000 is shared with other Isaac jobs, so sim renders at 1-2 Hz; the launch file relaxes
  face/frame timeouts in sim only.
- Arm geometry: camera azimuths on the subject's left (negative) keep joint 5 away from the wrist
  singularity; positive azimuths need large wrist flips.
- Since 2026-11-18 the camera is a BRIO held in a Robotiq 2F-85 (scripts/gripper.py; grip force 130,
  force 20 dropped it) and the seat is on base -y (`layout_yaw_deg: -90`, poses from
  scripts/gen_layout_poses.py). The arm node only reads the gripper status, never commands it.
- A finished sim run can leave arm/session/voice nodes alive; they answer the real session's service
  calls. Check `ros2 node list` before every real launch.
- The controller's dashboard port (29999) serves one client: no gripper.py/GetAngle while arm_node runs.

## Project goal

Nova5Photo is an onboarding portrait photographer for lab members:

- A **Dobot Nova 5** (6-DOF collaborative arm) carries a **webcam mounted on its flange**.
- The arm positions the camera around a seated/standing person to take portraits.
- **Audio guidance** tells the subject what to do (where to look, hold still, countdown, done).
- Captured images are **post-processed to add a depth-of-field (background blur) effect**, since a webcam cannot produce optical bokeh.

## Hardware and environment

- Lab machine: Ubuntu 22.04, NVIDIA RTX A4000.
- Robot control: **ROS 2 over TCP/IP** to the Nova 5 controller (Dobot TCP/IP protocol; ROS 2 Humble is the natural match for 22.04).
- Reference implementation for Nova 5 ROS 2 control: https://github.com/n7729697/TouchDrive/tree/codex/add-nova-ros2-mycobot — reuse its driver/bring-up approach instead of reinventing it.
- **Isaac Sim** is used to test motion and camera framing in simulation before running on the real arm. The first hardware test is planned for Monday 2026-11-16.

## Components

1. **Robot motion**: ROS 2 node that moves the arm through a set of safe, pre-validated camera poses relative to the subject; same interface for Isaac Sim and the real robot.
2. **Camera capture**: webcam driver/node (e.g. `usb_cam` or OpenCV) publishing images; hand-eye calibration between flange and camera.
3. **Subject framing**: face detection to centre/frame the subject before capture.
4. **Audio guidance**: text-to-speech or pre-recorded prompts driven by the capture state machine.
5. **Post-processing**: person segmentation or monocular depth estimation (GPU on the A4000) to blur the background and produce the final portrait.

## Safety notes

- The arm works close to people's faces. Keep speed/acceleration limits low, use collaborative/collision-detection modes, and validate every pose in Isaac Sim before it is used on hardware.
- Never send motion commands to the real robot from untested code paths; prefer a simulation flag that defaults to sim.
