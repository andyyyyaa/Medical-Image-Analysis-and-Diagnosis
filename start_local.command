#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "Starting Healthcare. Loading the models for the first time may take a while."
echo "Browser URL: http://127.0.0.1:${PORT:-5050}"
echo "Press Control+C in this terminal to stop the server."
exec .venv/bin/python start_local.py
