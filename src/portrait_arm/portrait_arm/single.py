"""Refuse to run a second copy of a node on the same ROS graph.

A finished Isaac Sim run once left its arm/session/voice nodes alive; they answered the real
session's service calls (the real arm moved while the session believed the move had failed).
"""
import time


def ensure_single(node, wait_s: float = 2.0):
    """Raise if another node with this node's name and namespace is visible (waits for discovery)."""
    me = (node.get_name(), node.get_namespace())
    t_end = time.time() + wait_s
    while time.time() < t_end:
        names = node.get_node_names_and_namespaces()
        if names.count(me) > 1:
            raise RuntimeError(
                f'another {me[1].rstrip("/")}/{me[0]} is already running on this ROS graph '
                '(an old sim or real launch?). Stop it first; check with `ros2 node list`.')
        time.sleep(0.2)
