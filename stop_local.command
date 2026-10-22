#!/bin/bash
set -e
cd "$(dirname "$0")"
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import shlex
import signal
import subprocess

pid_file = Path('.local/server.pid')
if not pid_file.exists():
    print('No local server is running.')
else:
    pid = int(pid_file.read_text().strip())
    command = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout.strip()
    args = shlex.split(command)
    expected = str(Path('start_local.py').resolve())
    if expected in args or 'start_local.py' in args:
        os.kill(pid, signal.SIGTERM)
        print('The local Healthcare server has been stopped.')
    else:
        print('The recorded server process has already exited.')
    pid_file.unlink(missing_ok=True)
PY
