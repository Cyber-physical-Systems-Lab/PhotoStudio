#!/usr/bin/env python3
"""Local neural text-to-speech service for the voice node (runs in .venv-tts, offline).

Loads every configured voice once and answers requests on a Unix socket, so each prompt is
synthesised in a fraction of a second instead of reloading a model per sentence. One JSON line per
request: {"lang": "en", "text": "...", "out": "/tmp/x.wav"} -> {"ok": true, "seconds": 0.12}.

Voices come from portrait.yaml (voice_node.voices), e.g.
    en: piper:voices/piper/en_GB-alba-medium.onnx
    sv: piper:voices/piper/sv_SE-nst-medium.onnx
    zh: kokoro:z:zm_yunxi                (Kokoro, lang code z = Mandarin, voice zm_yunxi)
Relative paths are relative to the workspace root. Started by portrait.launch.py (voice:=neural).
"""
import argparse
import json
import os
import socket
import sys
import threading
import time
import warnings
import wave

# Kokoro / PyTorch print harmless notices about how the model is built; keep the launch log clean.
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Piper:
    def __init__(self, model):
        from piper import PiperVoice
        self.v = PiperVoice.load(model)

    def synth(self, text, out):
        with wave.open(out, 'wb') as wf:
            self.v.synthesize_wav(text, wf)


class Kokoro:
    def __init__(self, lang_code, voice):
        from kokoro import KPipeline
        self.p = KPipeline(lang_code=lang_code, repo_id='hexgrad/Kokoro-82M', device='cpu')   # keep the shared GPU free
        self.voice = voice

    def synth(self, text, out):
        import numpy as np
        import soundfile as sf
        audio = np.concatenate([a for _, _, a in self.p(text, voice=self.voice)])
        sf.write(out, audio, 24000)


def load_voices(spec, log):
    voices = {}
    for lang, s in spec.items():
        kind, _, arg = s.partition(':')
        t0 = time.time()
        if kind == 'piper':
            path = arg if os.path.isabs(arg) else os.path.join(ROOT, arg)
            voices[lang] = Piper(path)
        elif kind == 'kokoro':
            code, _, name = arg.partition(':')
            voices[lang] = Kokoro(code, name)
        else:
            raise ValueError(f'unknown voice spec {s!r} for {lang}')
        voices[lang].synth('ok', '/tmp/.nova5photo_tts_warmup.wav')       # first call loads everything
        log(f'voice {lang}: {s} ready in {time.time() - t0:.1f} s')
    return voices


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=os.path.join(ROOT, 'src/portrait_bringup/config/portrait.yaml'))
    ap.add_argument('--socket', default='/tmp/nova5photo_tts.sock')
    a = ap.parse_args()
    log = lambda s: print(f'[tts] {s}', flush=True)  # noqa: E731
    vp = yaml.safe_load(open(a.config))['voice_node']['ros__parameters']
    voices = load_voices(vp['voices'], log)
    if os.path.exists(a.socket):
        os.unlink(a.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.socket)
    srv.listen(4)
    lock = threading.Lock()
    log(f'listening on {a.socket}')

    def handle(conn):
        with conn, conn.makefile('rw', encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    v = voices.get(r['lang']) or voices['en']
                    t0 = time.time()
                    with lock:
                        v.synth(r['text'], r['out'])
                    rep = {'ok': True, 'seconds': round(time.time() - t0, 3)}
                except Exception as e:  # noqa: BLE001
                    rep = {'ok': False, 'error': str(e)}
                f.write(json.dumps(rep) + '\n')
                f.flush()

    try:
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        if os.path.exists(a.socket):
            os.unlink(a.socket)


if __name__ == '__main__':
    sys.exit(main())
