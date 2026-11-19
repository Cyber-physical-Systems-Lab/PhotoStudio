"""ros2 run portrait_session start --name "Ada Lovelace" [--language sv]"""
import argparse

import rclpy
from portrait_interfaces.srv import StartSession


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--language', default='')
    args = ap.parse_args()
    rclpy.init()
    node = rclpy.create_node('portrait_start')
    cli = node.create_client(StartSession, '/portrait/start')
    if not cli.wait_for_service(timeout_sec=10.0):
        print('session node not running')
        return 1
    req = StartSession.Request()
    req.name, req.language = args.name, args.language
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
    res = fut.result()
    print(('started: ' if res.accepted else 'refused: ') + res.message)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if res.accepted else 1
