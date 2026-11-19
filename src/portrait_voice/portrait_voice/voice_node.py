"""Offline text-to-speech for the session.

Service /voice/say (portrait_interfaces/Say). With `key` set it speaks the phrase from the phrase
table in the configured language and also publishes the key on /portrait/guidance (the Isaac Sim
subject reacts to these keys). Every spoken sentence is published on /portrait/speech.

Engines: neural (default: scripts/tts_server.py keeps the Piper / Kokoro voices loaded; falls back
to espeak-ng if the service is not reachable), espeak-ng (always offline), none (log only).
"""
import json
import socket
import time
import os
import shutil
import subprocess
import tempfile
import threading

import rclpy
from portrait_interfaces.srv import Say
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from .phrases import phrase

ESPEAK_VOICES = {'en': 'en-gb', 'sv': 'sv', 'zh': 'cmn'}


class VoiceNode(Node):
    def __init__(self):
        super().__init__('voice_node')
        self.declare_parameter('language', 'en')
        self.declare_parameter('engine', 'espeak-ng')
        self.declare_parameter('rate_wpm', 150)
        self.declare_parameter('tts_socket', '/tmp/nova5photo_tts.sock')
        self.warned_fallback = False
        self.declare_parameter('subject_name', '')
        self.lock = threading.Lock()
        engine = self.get_parameter('engine').value
        if engine == 'neural':
            for _ in range(120):            # the service loads its models at launch (~10-30 s)
                if os.path.exists(self.get_parameter('tts_socket').value):
                    break
                time.sleep(0.5)
            else:
                self.get_logger().warn('neural TTS service not up yet; using espeak-ng until it is')
        if engine == 'espeak-ng' and not shutil.which('espeak-ng'):
            self.get_logger().warn('espeak-ng not found: voice will only be logged')
            engine = 'none'
        self.engine = engine
        self.speech_pub = self.create_publisher(String, '/portrait/speech', 10)
        self.guid_pub = self.create_publisher(String, '/portrait/guidance', 10)
        self.create_subscription(String, '/voice/set_language', self.on_lang, 10)
        self.create_subscription(String, '/voice/set_name', self.on_name, 10)
        self.create_service(Say, '/voice/say', self.on_say, callback_group=ReentrantCallbackGroup())
        self.get_logger().info(f'Voice engine: {self.engine}, language {self.lang}')

    @property
    def lang(self):
        return self.get_parameter('language').value

    def on_lang(self, msg):
        self.set_parameters([rclpy.parameter.Parameter('language', value=msg.data)])

    def on_name(self, msg):
        self.set_parameters([rclpy.parameter.Parameter('subject_name', value=msg.data)])

    def on_say(self, req, res):
        name = self.get_parameter('subject_name').value
        text = phrase(req.key, self.lang, name=name) if req.key else req.text
        if req.key:
            self.guid_pub.publish(String(data=req.key))
        self.speech_pub.publish(String(data=text))
        self.get_logger().info(f'SAY: {text}')
        if req.wait:
            self.speak(text)
        else:
            threading.Thread(target=self.speak, args=(text,), daemon=True).start()
        res.success = True
        return res

    def speak(self, text: str):
        with self.lock:
            try:
                if self.engine == 'neural' and self.speak_neural(text):
                    return
                if self.engine in ('espeak-ng', 'neural'):
                    subprocess.run(['espeak-ng', '-v', ESPEAK_VOICES.get(self.lang, 'en-gb'),
                                    '-s', str(self.get_parameter('rate_wpm').value), text],
                                   check=False, timeout=30, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                else:
                    time.sleep(0.06 * len(text.split()) + 0.3)   # rough speaking time
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'speech failed: {e}')

    def speak_neural(self, text: str) -> bool:
        """Synthesise with the TTS service and play it; False (-> espeak-ng) if that fails."""
        try:
            with tempfile.TemporaryDirectory() as d:
                wav = os.path.join(d, 'say.wav')
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(20)
                    s.connect(self.get_parameter('tts_socket').value)
                    s.sendall((json.dumps({'lang': self.lang, 'text': text, 'out': wav}) + '\n').encode())
                    rep = json.loads(s.makefile('r', encoding='utf-8').readline())
                if not rep.get('ok'):
                    raise RuntimeError(rep.get('error'))
                player = ['paplay', wav] if shutil.which('paplay') else ['aplay', '-q', wav]
                subprocess.run(player, check=False, timeout=60, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)      # paplay prints harmless X11 messages
            self.warned_fallback = False
            return True
        except Exception as e:  # noqa: BLE001
            if not self.warned_fallback:
                self.get_logger().warn(f'neural TTS unavailable ({e}); using espeak-ng')
                self.warned_fallback = True
            return False


def main():
    rclpy.init()
    node = VoiceNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
