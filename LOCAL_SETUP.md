# Running locally

Environment: macOS Apple Silicon, Python 3.12, and a project-local `.venv`.

## Start the server

Double-click `start_local.command`, wait for the models to load, then visit:

http://127.0.0.1:5050

You can also run the following in a terminal from the cloned project directory:

```bash
.venv/bin/python start_local.py
```

Press Control+C in the server terminal, or double-click `stop_local.command` to stop it.
If you use the background startup command in the README, logs are written to `.local/server.log`.
By default, the server listens only on the local machine. Port 5050 was chosen because macOS was already using port 5000 in the verified environment.
To change the port: `PORT=5051 .venv/bin/python start_local.py`.

## Reinstall the environment

Run from the project directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` records the full dependency versions used in the previously verified environment.
The heart prediction models were saved with scikit-learn 1.6.1, so load them with the same version.
Fracture inference uses CPU when CUDA is unavailable.
The first startup downloads the official base weights and caches them in `.local/torch`.
For a fresh clone from GitHub, download the models as described in the [README](README.md),
then run `.venv/bin/python init_database.py` to create an empty database.

The local session key is saved automatically in `.local/secret_key`. Do not share this file.
Existing `user_db.sqlite3` and model files can continue to be used.
Do not run `create_database.py`: it drops and recreates existing tables.
