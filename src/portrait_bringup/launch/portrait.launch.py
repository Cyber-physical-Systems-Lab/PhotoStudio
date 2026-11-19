"""One launch file for simulation and the real robot.

Simulation (default, safe):
    ros2 launch portrait_bringup portrait.launch.py
    ros2 launch portrait_bringup portrait.launch.py name:="Ada Lovelace" record:=/tmp/run
Real Nova 5 + webcam:
    ros2 launch portrait_bringup portrait.launch.py mode:=real robot_ip:=192.168.5.1 camera:=/dev/video0
Then start a session (if name:= was not given):
    ros2 run portrait_session start --name "Ada Lovelace"

Arguments
    mode        sim | real | fake      sim starts Isaac Sim; fake = kinematic arm, no camera (tests)
    name        start a session automatically for this person once everything is up
    language    en | sv
    robot_ip    controller IP (real)
    camera      webcam device (real)
    headless    true | false           Isaac Sim without a window (sim)
    record      directory for an mp4 of the Isaac Sim run (sim)
    isaac       true | false           start Isaac Sim from this launch (sim); false if already running
    voice       neural | espeak-ng | none
    rviz        true | false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def setup(context):
    a = {k: context.launch_configurations[k] for k in (
        'mode', 'name', 'language', 'robot_ip', 'camera', 'headless', 'record', 'isaac', 'voice',
        'rviz', 'config', 'isaac_args', 'layout')}
    mode = a['mode']
    if mode not in ('sim', 'real', 'fake'):
        raise RuntimeError(f'mode must be sim, real or fake (got {mode})')
    cfg = a['config'] or os.path.join(get_package_share_directory('portrait_bringup'), 'config', 'portrait.yaml')
    ws_root = os.environ.get('NOVA5PHOTO_ROOT', os.path.expanduser('~/Nova5Photo'))
    # Layout from scripts/calibrate_layout.py (seat, yaw, poses, limits) on top of portrait.yaml.
    layout = a['layout'] or os.path.join(ws_root, 'layout', 'current.yaml')
    if layout == 'none' or not os.path.exists(layout):
        if a['layout'] not in ('', 'none'):
            raise RuntimeError(f'layout file {layout} not found')
        layout = ''
    cfgs = [cfg] + ([layout] if layout else [])
    desc = get_package_share_directory('portrait_description')
    urdf_path = os.path.join(desc, 'urdf', 'nova5_portrait.urdf')
    with open(urdf_path) as f:
        urdf = f.read()

    acts = [LogInfo(msg=f'Nova5 portrait photographer: mode={mode}, config={cfg}, '
                        f'layout={layout or "portrait.yaml only"}')]
    if mode == 'real':
        acts.append(LogInfo(msg='REAL ROBOT: speed factor and limits from portrait.yaml. Keep the e-stop in reach.'))

    acts.append(Node(package='robot_state_publisher', executable='robot_state_publisher',
                     parameters=[{'robot_description': urdf}], output='log'))
    # Isaac Sim's path-traced camera runs at only a few frames per second on the lab GPU, so the
    # face watchdogs get more slack in simulation. The real webcam uses the values in portrait.yaml.
    sim_slack = {'safety.face_timeout': 3.0} if mode == 'sim' else {}
    acts.append(Node(package='portrait_arm', executable='arm_node', name='arm_node', output='screen',
                     parameters=cfgs + [{'backend': mode, 'urdf_path': urdf_path,
                                       'real.robot_ip': a['robot_ip'], **sim_slack}]))
    if mode == 'real':
        acts.append(Node(package='portrait_camera', executable='camera_node', name='camera_node',
                         output='screen', parameters=[cfg, {'device': a['camera']}]))
    if mode == 'sim' and a['isaac'] == 'true':
        cmd = [os.path.join(ws_root, 'sim', 'run_isaac.sh'), os.path.join(ws_root, 'sim', 'isaac_portrait_sim.py'),
               '--config', cfg] + (['--layout', layout] if layout else [])
        if a['headless'] == 'true':
            cmd.append('--headless')
        if a['record']:
            cmd += ['--record', a['record']]
        if a['isaac_args']:
            cmd += a['isaac_args'].split()
        isaac = ExecuteProcess(cmd=cmd, output='screen', name='isaac_sim')
        acts.append(isaac)
        # When Isaac exits (e.g. --exit-after-session) take every node down with it: leftover sim
        # nodes once answered a later real session's service calls.
        acts.append(RegisterEventHandler(OnProcessExit(target_action=isaac,
                                                       on_exit=[Shutdown(reason='Isaac Sim exited')])))

    acts.append(Node(package='portrait_perception', executable='perception_node', name='perception_node',
                     output='screen', parameters=[cfg]))
    engine = a['voice']
    if engine == 'neural':
        py = os.environ.get('PORTRAIT_TTS_PYTHON') or os.path.join(ws_root, '.venv-tts', 'bin', 'python')
        if os.path.exists(py):
            # Keep the neural models loaded between prompts.
            acts.append(ExecuteProcess(cmd=[py, os.path.join(ws_root, 'scripts', 'tts_server.py'), '--config', cfg],
                                       output='screen', name='tts_server'))
        else:
            acts.append(LogInfo(msg=f'{py} not found: using espeak-ng (see README, voices)'))
            engine = 'espeak-ng'
    acts.append(Node(package='portrait_voice', executable='voice_node', name='voice_node', output='screen',
                     parameters=[cfg, {'language': a['language'], 'engine': engine}]))
    session_params = {'language': a['language']}
    if mode == 'sim':
        session_params.update({'face_lost_timeout_s': 3.0, 'settle_s': 2.5, 'frame_timeout_s': 8.0})
    if a['name']:
        session_params['auto_start_name'] = a['name']
    acts.append(Node(package='portrait_session', executable='session_node', name='session_node',
                     output='screen', parameters=cfgs + [session_params]))
    if a['rviz'] == 'true':
        acts.append(Node(package='rviz2', executable='rviz2', output='log'))
    return acts


def generate_launch_description():
    args = [
        DeclareLaunchArgument('mode', default_value='sim'),
        DeclareLaunchArgument('name', default_value=''),
        DeclareLaunchArgument('language', default_value='en'),
        DeclareLaunchArgument('robot_ip', default_value='192.168.0.37'),
        DeclareLaunchArgument('camera', default_value='/dev/video0'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('record', default_value=''),
        DeclareLaunchArgument('isaac', default_value='true'),
        DeclareLaunchArgument('isaac_args', default_value=''),
        DeclareLaunchArgument('voice', default_value='neural', description='neural | espeak-ng | none'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('config', default_value=''),
        DeclareLaunchArgument('layout', default_value='',
                              description='layout file from calibrate_layout.py (default: layout/current.yaml '
                                          'if it exists; "none" = portrait.yaml only)'),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=setup)])
