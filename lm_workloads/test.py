import os
import signal

pid = 3452705  # 目标进程的 PID
os.kill(pid, signal.SIGUSR2)