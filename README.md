# Healthcare 6.0 · ClinRisk

A health assessment web application built with Flask, PyTorch, and scikit-learn. It provides fracture image analysis, heart disease risk prediction, account management, and record searches. The interface is in English.

## Features

- **Fracture image analysis**: Upload an image, detect and draw bounding boxes with a multitask Faster R-CNN, and classify fractures with ResNet50. Supports 10 fracture types.
- **Heart disease risk prediction**: Use logistic regression or a Perceptron to predict risk from inputs such as age, blood pressure, and cholesterol.
- **Accounts and records**: Sign up, log in, log out, and search or filter assessment records.
- **Statistics**: View age, gender, and fracture type distributions in the records.

This project is intended for learning and demonstration. Predictions cannot replace clinical diagnosis.

## Technology stack

| Layer | Technologies |
| --- | --- |
| Web backend | Flask, Flask-CORS, Jinja2 |
| Imaging models | PyTorch 2.6.0, torchvision 0.21.0 |
| Heart prediction | scikit-learn 1.6.1 |
| Data processing and charts | pandas, NumPy, Matplotlib, Seaborn |
| Database | SQLite |
| Frontend | Bootstrap, HTML, CSS, JavaScript |

The application was previously verified to start on macOS Apple Silicon with Python 3.12. The local startup script uses `fcntl` and supports macOS / Linux; Windows users can use WSL. The current code uses CUDA when available and CPU otherwise. Apple MPS is not enabled.

## Quick start

### 1. Get the source and install dependencies

```bash
git clone https://github.com/andyyyyaa/Medical-Image-Analysis-and-Diagnosis.git
cd Medical-Image-Analysis-and-Diagnosis
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Download the models

Models are distributed as [models-v1 release assets](https://github.com/andyyyyaa/healthcare_6.0/releases/tag/models-v1) and are not stored in Git history. Download these four files and place them in the project root:

| File | Purpose |
| --- | --- |
| `best_multitask_frcnn_verbose3.pt` | Fracture detection and global classification |
| `model.pth` | ResNet50 fracture type classification |
| `heart_model_lr.pkl` | Heart disease logistic regression model |
| `heart_model_p.pkl` | Heart disease Perceptron model |

Alternatively, use GitHub CLI while signed in with access to the model repository:

```bash
gh release download models-v1 --repo andyyyyaa/healthcare_6.0 \
  --pattern '*.pt' --pattern '*.pth' --pattern '*.pkl' --pattern 'SHA256SUMS' --dir .
shasum -a 256 -c SHA256SUMS
```

Load model files only from trusted sources. Use scikit-learn 1.6.1 to match the version used to save the heart models. On the first startup, torchvision also downloads the official Faster R-CNN base weights and caches them in `.local/torch`, so an internet connection is required.

### 3. Initialize the database

```bash
python init_database.py
```

This command creates missing tables only. It can be run repeatedly without deleting existing records. The repository does not contain existing accounts or diagnosis data; register an account in a new environment.

The legacy `create_database.py` script drops and recreates tables. Do not run it when you have existing data.

### 4. Start the server

```bash
python start_local.py
```

Wait for the models to load, then open **http://127.0.0.1:5050** in a browser. Press `Control+C` to stop the server.

After installing dependencies, downloading the models, and initializing the database, macOS users can also double-click `start_local.command` to start the server and `stop_local.command` to stop it.

To change the port:

```bash
PORT=5051 python start_local.py
```

To run in the background and save logs:

```bash
mkdir -p .local
nohup .venv/bin/python -u start_local.py > .local/server.log 2>&1 &
```

## Project structure

```text
Medical-Image-Analysis-and-Diagnosis/
├── app.py                    # Flask routes and model inference
├── start_local.py            # Local startup, session key, and cache configuration
├── start_server.py           # Server startup entry point
├── start_local.command       # macOS launcher for the local server
├── stop_local.command        # macOS script to stop the local server
├── init_database.py          # Create missing tables without deleting records
├── create_database.py        # Legacy database reset script (deletes data)
├── train_model.py            # Heart prediction model training
├── infer-fracture.py         # Standalone fracture inference
├── templates/                # Jinja2 page templates
├── static/                   # Styles, scripts, public images, and frontend dependencies
├── requirements.txt          # Python dependency versions
├── LOCAL_SETUP.md            # Local setup instructions
└── DEPLOYMENT.md             # Server deployment reference
```

## Data and configuration

- `user_db.sqlite3` stores accounts and assessment records locally and is not committed to GitHub.
- `.local/secret_key` is generated automatically by the local startup script and reused across runs. You can also set the `SECRET_KEY` environment variable.
- Virtual environments, logs, caches, generated annotation images, and statistical charts are excluded by `.gitignore`.
- Local test images and `static/heart.csv` are not included. Training data is not required to start the application or use existing models. To run `train_model.py`, provide a CSV with the columns `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`.
- The current account code uses plaintext passwords and includes debug logs that print login information. This version is suitable for local demonstrations. Before exposing it publicly, add password hashing, redact logs, improve access control, and configure HTTPS. The [deployment guide](DEPLOYMENT.md) is a configuration reference; it does not mean production hardening has been completed.

## Troubleshooting

**Model file not found**: Confirm that the four release assets are in the same directory as `app.py` and that their filenames have not changed.

**`no such table` error**: Run `python init_database.py` in the virtual environment.

**Port already in use**: macOS may use port 5000 for system services. The local default is 5050; use `PORT` to change it.

**Slow first startup or base weight download failure**: The first startup downloads and loads torchvision weights. Check your connection and retry. Weights are cached in `.local/torch`.

**`unexpected keys: ['cls_loss_fn.pos_weight']` message**: Local startup reports this extra checkpoint field. The models were previously verified to load and the server to start despite this message. This does not validate model accuracy or clinical effectiveness.
