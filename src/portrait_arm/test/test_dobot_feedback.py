import numpy as np

from portrait_arm.dobot_v3 import FEEDBACK_DTYPE, FEEDBACK_MAGIC


def test_layout_matches_dobot_v3():
    assert FEEDBACK_DTYPE.itemsize == 1440
    f = FEEDBACK_DTYPE.fields
    assert f['robot_mode'][1] == 24
    assert f['test_value'][1] == 48
    assert f['q_actual'][1] == 432
    assert f['tool_vector_actual'][1] == 624


def test_parse_packet():
    pkt = np.zeros(1, FEEDBACK_DTYPE)
    pkt['len'] = 1440
    pkt['test_value'] = FEEDBACK_MAGIC
    pkt['robot_mode'] = 5
    pkt['q_actual'] = [10, 20, 30, 40, 50, 60]
    fb = np.frombuffer(pkt.tobytes(), FEEDBACK_DTYPE)[0]
    assert int(fb['robot_mode']) == 5
    assert list(fb['q_actual']) == [10, 20, 30, 40, 50, 60]


def test_feedback_reconnects_after_drop():
    """Fake controller: stream, drop the connection, accept again -> feedback recovers by itself."""
    import socket
    import threading
    import time

    from portrait_arm.dobot_v3 import DobotV3Client

    pkt = np.zeros(1, FEEDBACK_DTYPE)
    pkt['len'] = 1440
    pkt['test_value'] = FEEDBACK_MAGIC
    pkt['robot_mode'] = 5
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', 0))
    srv.listen(2)
    port = srv.getsockname()[1]
    stop = threading.Event()

    def serve():
        for session, q in enumerate((1.0, 2.0)):
            conn, _ = srv.accept()
            pkt['q_actual'] = [q] * 6
            t0 = time.time()
            while time.time() - t0 < 0.6 and not stop.is_set():
                try:
                    conn.sendall(pkt.tobytes())
                except OSError:
                    break
                time.sleep(0.008)
            conn.close()                      # the drop
            if session == 1:
                while not stop.is_set():
                    time.sleep(0.05)

    threading.Thread(target=serve, daemon=True).start()
    warnings = []
    c = DobotV3Client('127.0.0.1', timeout=1.0, log=warnings.append, warn=warnings.append)
    c.fb_port = port
    c._fb_sock = socket.create_connection(('127.0.0.1', port), timeout=1.0)
    c._running = True
    threading.Thread(target=c._feedback_loop, daemon=True).start()
    try:
        deadline = time.time() + 5
        seen = set()
        while time.time() < deadline and seen != {1.0, 2.0}:
            fb = c.feedback()
            if fb is not None:
                seen.add(float(fb['q_actual'][0]))
            time.sleep(0.01)
        assert seen == {1.0, 2.0}, (seen, warnings)
        assert any('reconnected' in w for w in warnings), warnings
    finally:
        stop.set()
        c._running = False
        srv.close()
