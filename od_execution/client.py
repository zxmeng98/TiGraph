import os
import argparse
import signal

signal_pause = signal.SIGUSR1
signal_resume = signal.SIGUSR2

def send_signal(pid, action):
    if action == 'pause':
        os.kill(pid, signal_pause)
    elif action == 'resume':
        os.kill(pid, signal_resume)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, default=0)
    parser.add_argument('--action', type=str, default='pause')
    args = parser.parse_args()
    send_signal(args.pid, args.action)
