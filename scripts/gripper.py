#!/usr/bin/env python3
"""Robotiq 2F-85 on the Nova 5 tool connector (Modbus RTU through the controller's dashboard port).

Run it only while the portrait launch is STOPPED (the controller's dashboard port 29999 serves one
client at a time) and with the robot enabled.

    scripts/gripper.py status
    scripts/gripper.py activate              # first use after power-up: fingers do a full open/close stroke
    scripts/gripper.py grasp [--force 130]   # close slowly on the camera, check it is held
    scripts/gripper.py release --yes         # OPENS the fingers: hold the camera first or it falls
"""
import argparse
import re
import socket
import sys
import time

IP = '192.168.0.37'


class Dash:
    def __init__(self, ip):
        self.s = socket.create_connection((ip, 29999), timeout=5)

    def __call__(self, cmd):
        self.s.sendall(cmd.encode())
        buf = b''
        while not buf.rstrip().endswith(b';'):
            chunk = self.s.recv(1024)
            if not chunk:
                sys.exit('controller closed the connection (is the portrait launch still running?)')
            buf += chunk
        reply = buf.decode().strip()
        if not reply.startswith('0,'):
            sys.exit(f'{cmd} failed: {reply}')
        return reply


def regs(reply):
    return [int(x) for x in re.search(r'\{(.*?)\}', reply).group(1).split(',') if x.strip()]


def status(d, idx):
    v = regs(d(f'GetInRegs({idx},2000,3)'))
    st = v[0] >> 8
    return {'activated': bool(st & 1), 'ready': (st >> 4) & 3 == 3,
            'object': {0: 'moving', 1: 'contact while opening', 2: 'HOLDING OBJECT', 3: 'no object'}[(st >> 6) & 3],
            'fault': hex(v[1] >> 8), 'position': v[2] >> 8}


def go(d, idx, pos, speed, force):
    d(f'SetHoldRegs({idx},1000,3,{{2304,{pos},{(speed << 8) | force}}})')
    time.sleep(0.5)
    for _ in range(40):
        s = status(d, idx)
        if s['object'] != 'moving':
            return s
        time.sleep(0.25)
    return status(d, idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['status', 'activate', 'grasp', 'release'])
    ap.add_argument('--ip', default=IP)
    ap.add_argument('--force', type=int, default=130, help='0-255 (130 ~ 130 N; 20 was too weak for the BRIO)')
    ap.add_argument('--speed', type=int, default=30, help='0-255')
    ap.add_argument('--yes', action='store_true', help='required for release')
    a = ap.parse_args()
    d = Dash(a.ip)
    idx = int(re.search(r'\{(\d+)\}', d('ModbusCreate(127.0.0.1,60000,9,1)')).group(1))
    try:
        if a.action == 'activate':
            d(f'SetHoldRegs({idx},1000,3,{{0,0,0}})'); time.sleep(0.5)
            d(f'SetHoldRegs({idx},1000,3,{{256,0,0}})')
            for _ in range(40):
                time.sleep(0.25)
                if status(d, idx)['ready']:
                    break
        elif a.action == 'grasp':
            s = go(d, idx, 255, a.speed, a.force)
            print(s)
            time.sleep(2.0)
            s = status(d, idx)
            print('after 2 s:', s)
            sys.exit(0 if s['object'] == 'HOLDING OBJECT' else 'NOT holding the camera: place it and grasp again')
        elif a.action == 'release':
            if not a.yes:
                sys.exit('release opens the fingers and drops the camera; hold it and add --yes')
            go(d, idx, 0, 40, 50)
        print(status(d, idx))
    finally:
        try:
            d(f'ModbusClose({idx})')
        except SystemExit:
            pass


if __name__ == '__main__':
    main()
