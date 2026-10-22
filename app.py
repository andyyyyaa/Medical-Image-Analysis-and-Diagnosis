#set up virtual environment:
#cd healthcare_3.0
#python3 -m venv .venv --prompt slai-web-pub-main
#source .venv/bin/activate
#pip install -r requirement.txt

import sqlite3
from flask import Flask, request,session, render_template, abort, flash, redirect, url_for, get_flashed_messages
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import io
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import seaborn as sns
import uuid

matplotlib.use('Agg')

# Create the Flask application
app = Flask(__name__)

# Production configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['SESSION_COOKIE_SECURE'] = False  # Set to False for HTTP and True for HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access to session cookies
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # Session lifetime: one hour

# CORS configuration
CORS(app, resources={r"/*": {"origins": "*"}})

# Set the maximum upload size (16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


fracture_id = {'name':[''],'sex':[''],'age':[''],'classification':['']}

def _remove_bbox_file(filename: str):
    if not filename:
        return
    try:
        path = os.path.join(app.static_folder, filename)
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


@app.before_request
def cleanup_bbox_on_navigate():
    if request.endpoint in (None, "static"):
        return
    if request.endpoint == "upload" and request.method == "POST":
        old_bbox = session.pop("bbox_image", None)
        if old_bbox:
            _remove_bbox_file(old_bbox)
        return
    old_bbox = session.pop("bbox_image", None)
    if old_bbox:
        _remove_bbox_file(old_bbox)


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return f'{pct:.1f}%\n({val})'
    return my_autopct

def filestorage_to_image(file_storage):
    file_content = file_storage.read()
    image_stream = io.BytesIO(file_content)
    image = Image.open(image_stream)
    return image

def query(form: dict):
    conn = sqlite3.connect("user_db.sqlite3")
    sql = "SELECT * FROM fracture WHERE 1=1"
    params = []
    if form["sex"] != "Any":
        sql += " AND sex = ?"
        params.append(form["sex"])
    if form["classification"] != "Any":
        sql += " AND fracture = ?"
        params.append(form["classification"])
    if form["min_age"] != "":
        sql += " AND age >= ?"
        params.append(int(form["min_age"]))
    
    if form["max_age"] != "":
        sql += " AND age <= ?"
        params.append(int(form["max_age"]))
    
    if form["name"] != "":
        sql += " AND name = ?"
        params.append(form["name"])

    # Execute the query and load the results
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()

    # Return the original list[dict] format
    return df.to_dict(orient="records")



def query_heart(form: dict):
    conn = sqlite3.connect("user_db.sqlite3")
    sql = "SELECT * FROM heart WHERE 1=1"
    params = []
    if form["sex"] != "Any":
        sql += " AND sex = ?"
        params.append(form["sex"])
    if form["target"] != "Any":
        sql += " AND target = ?"
        params.append(form["classification"])
    if form["min_age"] != "":
        sql += " AND age >= ?"
        params.append(int(form["min_age"]))
    
    if form["max_age"] != "":
        sql += " AND age <= ?"
        params.append(int(form["max_age"]))
    
    if form["name"] != "":
        sql += " AND name = ?"
        params.append(form["name"])

    # Execute the query and load the results
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()

    # Return the original list[dict] format
    return df.to_dict(orient="records")





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRACTURE_DET_IMG_SIZE = 336
FRACTURE_DET_SCORE_THR = 0.3
FRACTURE_GLOBAL_THR = 0.5
FRACTURE_DET_MODEL_PATH = "best_multitask_frcnn_verbose3.pt"


def preprocess_keep_aspect_pad(img: Image.Image, out_size: int = 336):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(out_size / w, out_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

    pad_w = out_size - new_w
    pad_h = out_size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    pad_right = pad_w - pad_left
    pad_bottom = pad_h - pad_top

    img_t = F.to_tensor(img_resized)
    img_t = F.pad(img_t, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

    meta = {
        "orig_w": w,
        "orig_h": h,
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
    }
    return img_t, meta


def map_boxes_to_original(boxes_xyxy_336: np.ndarray, meta: dict) -> np.ndarray:
    scale = float(meta["scale"])
    pad_left = float(meta["pad_left"])
    pad_top = float(meta["pad_top"])
    orig_w = float(meta["orig_w"])
    orig_h = float(meta["orig_h"])

    b = boxes_xyxy_336.astype(np.float32).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - pad_left) / max(scale, 1e-8)
    b[:, [1, 3]] = (b[:, [1, 3]] - pad_top) / max(scale, 1e-8)

    b[:, 0] = np.clip(b[:, 0], 0, orig_w - 1)
    b[:, 1] = np.clip(b[:, 1], 0, orig_h - 1)
    b[:, 2] = np.clip(b[:, 2], 0, orig_w - 1)
    b[:, 3] = np.clip(b[:, 3], 0, orig_h - 1)

    keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    return b[keep]


def map_detections_to_original(
    boxes_336: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    meta: dict
):
    mapped = []
    mapped_scores = []
    mapped_labels = []
    for b, s, lab in zip(boxes_336, scores, labels):
        bb = map_boxes_to_original(b[None, :], meta)
        if bb.shape[0] == 0:
            continue
        mapped.append(bb[0])
        mapped_scores.append(float(s))
        mapped_labels.append(int(lab))

    if len(mapped) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    return (
        np.stack(mapped, axis=0).astype(np.float32),
        np.array(mapped_scores, dtype=np.float32),
        np.array(mapped_labels, dtype=np.int32),
    )


def draw_boxes_on_image(
    img: Image.Image,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    score_thr: float = 0.3
) -> Image.Image:
    img_out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img_out)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), s in zip(boxes_xyxy, scores):
        if float(s) < score_thr:
            continue
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

        text = f"{float(s):.3f}"
        tx, ty = x1, max(0, y1 - 18)
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        draw.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill=(255, 0, 0))
        draw.text((tx + 3, ty + 2), text, fill=(255, 255, 255), font=font)

    return img_out


class MultiTaskFasterRCNN(nn.Module):
    def __init__(self, img_size: int = 336):
        super().__init__()

        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

        in_features = detector.roi_heads.box_predictor.cls_score.in_features
        detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

        detector.transform.min_size = (img_size,)
        detector.transform.max_size = img_size

        detector.rpn.pre_nms_top_n_train = 1000
        detector.rpn.post_nms_top_n_train = 1000
        detector.rpn.pre_nms_top_n_test = 1000
        detector.rpn.post_nms_top_n_test = 300
        detector.roi_heads.batch_size_per_image = 128
        detector.roi_heads.detections_per_img = 100

        self.detector = detector
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)
        )

    def forward(self, images):
        det_out = self.detector(images)
        imgs_t, _ = self.detector.transform(images, None)
        feats = self.detector.backbone(imgs_t.tensors)
        feat = feats["pool"] if ("pool" in feats) else list(feats.values())[-1]
        logits = self.cls_head(feat)
        probs = torch.sigmoid(logits).squeeze(1)
        return det_out, probs


fracture_detection_model = MultiTaskFasterRCNN(img_size=FRACTURE_DET_IMG_SIZE)
ckpt = torch.load(FRACTURE_DET_MODEL_PATH, map_location="cpu")
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
missing_keys, unexpected_keys = fracture_detection_model.load_state_dict(state, strict=False)
if len(missing_keys) > 0 or len(unexpected_keys) > 0:
    print("[WARN] fracture detection load_state_dict not strict.")
    print("  missing keys:", missing_keys[:10], "..." if len(missing_keys) > 10 else "")
    print("  unexpected keys:", unexpected_keys[:10], "..." if len(unexpected_keys) > 10 else "")
fracture_detection_model = fracture_detection_model.to(device)
fracture_detection_model.eval()

fracture_classification_model = models.resnet50() 
fracture_classification_model.fc = nn.Linear(fracture_classification_model.fc.in_features, 10)
fracture_classification_model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=True))
fracture_classification_model = fracture_classification_model.to(device)
fracture_classification_model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class_labels = ['Avulsion Fracture', 'Comminuted Fracture', 'Fracture Dislocation', 'Greenstick Fracture', 'Hairline Fracture', 'Impacted Fracture', 'Longitudinal Fracture', 'Oblique Fracture', 'Pathological Fracture', 'Spiral Fracture']  


fracture_treatment = {
    "Avulsion Fracture": "Stop activity immediately, avoid pulling the injured area. Apply ice to reduce swelling, immobilize the fracture site, and seek medical attention.",
    "Comminuted Fracture": "Avoid any movement due to multiple bone fragments. Use cold compresses to reduce swelling, stabilize the fracture with a bandage, and seek immediate medical attention.",
    "Fracture Dislocation": "Do not attempt to reposition by yourself or apply pressure. Immobilize with splint or bandage and seek immediate medical treatment, usually requiring surgical intervention.",
    "Greenstick Fracture": "Common in children, causing partial bending without complete break. Reduce pain by supporting and immobilizing the fracture. Seek medical attention, possibly requiring cast or splint.",
    "Hairline Fracture": "Usually minor but requires rest; avoid putting weight on the area. Seek medical evaluation, may require splint or cast.",
    "Impacted Fracture": "Bone ends impacted into each other; avoid forced movement or weight-bearing. Apply ice to reduce swelling, immobilize and seek medical care, typically requiring imaging to determine surgical needs.",
    "Longitudinal Fracture": "Fracture occurs lengthwise along the bone. Avoid movement immediately, stabilize using a splint or bandage, and seek prompt medical attention.",
    "Oblique Fracture": "Typically an angled fracture; immediately avoid moving the area, immobilize with a splint or brace, and promptly seek medical care.",
    "Pathological Fracture": "Occurs due to underlying conditions such as tumors or osteoporosis. Avoid pressure, limit activity, and seek medical attention immediately for thorough evaluation and treatment.",
    "Spiral Fracture": "Caused by twisting or pulling force; avoid weight-bearing immediately, immobilize the fracture, and seek medical attention promptly, usually requiring imaging to assess injury severity."
}
@app.route('/')
def home():
    session.setdefault("login_name", "")
    if session['login_name']!="":
        return render_template('home.html',login_name=session['login_name'])
    else:
        return render_template('home.html',login_name=None)

# @app.route('/reset')
# def reset_data():
#     reset_dataset()
#     return redirect('/')



@app.route('/diagnosis')
def diagnosis():
    return render_template('diagnosis.html')


@app.post('/upload')
def upload():
    

    if not (request.files.get("file_name")):
        flash("No file input!","error")
        return redirect('/diagnosis')
    img = filestorage_to_image(request.files.get("file_name")).convert("RGB")

    # image = Image.open(io.BytesIO(the_file))
    input_tensor = preprocess(img).unsqueeze(0)  
    input_tensor = input_tensor.to(device)

    det_img_t, det_meta = preprocess_keep_aspect_pad(img, out_size=FRACTURE_DET_IMG_SIZE)
    det_images = [det_img_t.to(device)]
    use_amp = device.type == "cuda"

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            det_out, global_probs = fracture_detection_model(det_images)

    pred = det_out[0]
    global_prob = float(global_probs[0].detach().cpu().item())
    boxes_336 = pred["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = pred["scores"].detach().cpu().numpy().astype(np.float32)
    labels = pred.get("labels", torch.zeros((len(scores),), dtype=torch.int64)).detach().cpu().numpy()

    boxes_orig, scores_orig, labels_orig = map_detections_to_original(boxes_336, scores, labels, det_meta)
    max_det_score = float(scores_orig.max()) if scores_orig.size > 0 else 0.0
    has_fracture = (max_det_score >= FRACTURE_DET_SCORE_THR) or (global_prob >= FRACTURE_GLOBAL_THR)

    vis = draw_boxes_on_image(img, boxes_orig, scores_orig, score_thr=FRACTURE_DET_SCORE_THR)
    os.makedirs(app.static_folder, exist_ok=True)
    bbox_filename = f"fracture_bbox_{uuid.uuid4().hex}.jpg"
    bbox_path = os.path.join(app.static_folder, bbox_filename)
    vis.save(bbox_path)
    bbox_image_url = url_for("static", filename=bbox_filename)
    session["bbox_image"] = bbox_filename
    global_confidence = f"{global_prob * 100:.2f}%"
    detection_confidence = (
        f"{max_det_score * 100:.2f}%"
        if scores_orig.size > 0
        else global_confidence
    )

    if not has_fracture:
        result = {
            'classification': 'No Fracture',
            'confidence': global_confidence,
            'advice': 'No fracture, please keep health',
        }
    else:
        with torch.no_grad():
            output = fracture_classification_model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, 1)
            predicted_class = class_labels[predicted_class_idx.item()]
            confidence_percentage = confidence.item() * 100

        result = {
            'classification': predicted_class,
            'confidence': f'{confidence_percentage:.2f}%',
            'advice': fracture_treatment[predicted_class],
        }
    if(session.get('login_name')!=("" or None)):
        conn = sqlite3.connect("user_db.sqlite3")
        fracture_df = pd.read_sql_query("SELECT * FROM fracture", conn)
        fracture_df.loc[fracture_df['name'] == session.get('login_name'),'fracture']=result["classification"]
        # name_index = fracture_df[fracture_df['name'] == session['login_name']]
        # name_index.iloc[0]['classification']=result["classification"]
        #fracture_df.drop(index=fracture_df[fracture_df['name'] == session['login_name']])
        #name_index.to_csv("fracture_data.csv",mode='a',index=False,header=False)
        #fracture_df.to_csv("fracture_data.csv",index=False)
        fracture_df.to_sql('fracture', conn, if_exists='replace', index=False)
        conn.close()
    return render_template(
        'result.html',
        classification=result["classification"],
        confidence=result["confidence"],
        advice=result["advice"],
        bbox_image_url=bbox_image_url,
        detection_confidence=detection_confidence
    )

@app.route("/heart_disease_prediction")
def heart_disease_prediction():
    if session.get('login_name') and session.get('login_name') != "":
        login_status = True
    else:
        login_status = False
    print(f"login_status: {login_status}")
    return render_template("heart.html", login_status=login_status)

@app.post("/heart_disease_prediction_result")
def heart_disease_prediction_result():
    with open("heart_model_lr.pkl", "rb") as f:
        heart_model_lr = pickle.load(f)
    with open("heart_model_p.pkl", "rb") as f:
        conn = sqlite3.connect("user_db.sqlite3")
        heart_model_p = pickle.load(f)    
        input_age=request.form.get("age")
        sex=request.form.get("sex")
        cp=request.form.get("cp")
        input_trestbps=request.form.get("trestbps")
        input_chol=request.form.get("chol")
        fbs=request.form.get("fbs")
        restecg=request.form.get("restecg")
        input_thalach=request.form.get("thalach")
        exang=request.form.get("exang")
        input_oldpeak=request.form.get("oldpeak")
        blank=True
        if session.get('login_name') and session.get('login_name') != "":
            login_status = True
            raw_df = pd.read_sql_query("SELECT * FROM heart", conn)
            df = raw_df[raw_df['name'] == session.get('login_name')]
            sex = df['sex'].iloc[0]
            input_age = df['age'].iloc[0]
        else:
            login_status = False

        if input_age:
            if sex:
                if cp:
                    if input_trestbps:
                        if input_chol:
                            if fbs:
                                if restecg:
                                    if input_thalach:
                                        if exang:
                                            if input_oldpeak:
                                                blank=False
        if blank==True:
            flash("not valid input")
            return redirect(url_for("heart_disease_prediction"))




        if sex == "male":
            input_sex_0=0
            input_sex_1=1
        else:#female
            input_sex_0=1
            input_sex_1=0
        
        if cp=="non-anginal":
            input_cp_0=1
            input_cp_1=0
            input_cp_2=0
            input_cp_3=0
        elif cp=="typical angina":
            input_cp_0=0
            input_cp_1=1
            input_cp_2=0
            input_cp_3=0
        elif cp=="atypical angina":
            input_cp_0=0
            input_cp_1=0
            input_cp_2=1
            input_cp_3=0
        else:#asymptomatic
            input_cp_0=0
            input_cp_1=0
            input_cp_2=0
            input_cp_3=1

        if int(fbs)>120:#Above 120 mg/dl
            input_fbs_0=0
            input_fbs_1=1
        else:#120 mg/dl or below
            input_fbs_0=1
            input_fbs_1=0

        if restecg=="normal":
            input_restecg_0=1
            input_restecg_1=0
            input_restecg_2=0
        elif restecg=="ST-T wave abnormality":
            input_restecg_0=0
            input_restecg_1=1
            input_restecg_2=0
        else:#probable or definite left ventricular hypertrophy 
            input_restecg_0=0
            input_restecg_1=0
            input_restecg_2=1

        if exang=="Exercise-induced angina True":
            input_exang_0=0
            input_exang_1=1
        else:#Exercise-induced angina False
            input_exang_0=1
            input_exang_1=0

        input_data = np.array([[int(input_age), input_sex_0, input_sex_1, input_cp_0, input_cp_1, input_cp_2, input_cp_3, int(input_trestbps), int(input_chol), input_fbs_0, input_fbs_1, input_restecg_0, input_restecg_1, input_restecg_2, int(input_thalach), input_exang_0, input_exang_1, float(input_oldpeak)]])
        
        model_selection=request.form.get("model_selection")
        if model_selection=="lr":
            # Predict probabilities using logistic regression
            pred_proba = heart_model_lr.predict_proba(input_data)
            pred = heart_model_lr.predict(input_data)
            model_selection_name="Logistic Regression 1.0"
            # Get the predicted heart disease probability
            heart_disease_probability = pred_proba[0][1] * 100  # Convert to a percentage
        elif model_selection=="p":
            # The Perceptron predicts classes only, without probabilities
            pred = heart_model_p.predict(input_data)
            model_selection_name="Perceptron 1.0"
            heart_disease_probability = None  # The Perceptron does not provide probabilities
        
        if pred[0]==0:
            prediction=False
            prediction_text="no"
        else:#pred[0]==1
            prediction=True
            prediction_text="yes"
            
        if(session.get('login_name')!=("" or None)):
            conn = sqlite3.connect("user_db.sqlite3")
            heart_df = pd.read_sql_query("SELECT * FROM heart", conn)
            heart_df.loc[heart_df['name'] == session.get('login_name'),'target']=prediction_text
            heart_df.to_sql('heart', conn, if_exists='replace', index=False)
            
        from datetime import datetime
        
        return render_template("heart_result.html",
                             prediction=prediction,
                             prediction_text=prediction_text,
                             model_selection=model_selection,
                             model_selection_name=model_selection_name,
                             heart_disease_probability=heart_disease_probability,
                             current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@app.get("/search")
def search():
    # Check whether the user is logged in
    login_status = False
    user_recent_records = []
    
    if session.get('login_name') and session.get('login_name') != "":
        login_status = True
        # Query the current user's five most recent fracture records
        conn = sqlite3.connect("user_db.sqlite3")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name, sex, age, fracture, 
                   datetime('now', 'localtime') as current_time
            FROM fracture 
            WHERE name = ? AND fracture != '' AND fracture != 'No Fracture'
            ORDER BY rowid DESC 
            LIMIT 5
        """, (session['login_name'],))
        user_recent_records = cursor.fetchall()
        conn.close()
    
    # Get the available filter options
    conn = sqlite3.connect("user_db.sqlite3")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT sex FROM fracture")
    sex_types = set(row[0] for row in cursor.fetchall())
    cursor.execute("SELECT DISTINCT fracture FROM fracture")
    classification_types = set(row[0] for row in cursor.fetchall())
    conn.close()
    
    return render_template("search.html",
                         sex_types=sex_types,
                         classification_types=classification_types,
                         name=session.get('login_name'),
                         login_status=login_status,
                         user_recent_records=user_recent_records)

@app.post("/table")
def table():
    return render_template("table.html",data=query(dict(request.form)))

@app.get("/search_heart")
def search_heart():
    # Check whether the user is logged in
    login_status = False
    user_recent_records = []
    
    if session.get('login_name') and session.get('login_name') != "":
        login_status = True
        # Query the current user's five most recent heart assessment records
        conn = sqlite3.connect("user_db.sqlite3")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name, sex, age, target, 
                   datetime('now', 'localtime') as current_time
            FROM heart 
            WHERE name = ? AND target != ''
            ORDER BY rowid DESC 
            LIMIT 5
        """, (session['login_name'],))
        user_recent_records = cursor.fetchall()
        conn.close()
    
    # Get the available filter options
    conn = sqlite3.connect("user_db.sqlite3")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT sex FROM heart")
    sex_types = set(row[0] for row in cursor.fetchall())
    cursor.execute("SELECT DISTINCT target FROM heart")
    target_types = set(row[0] for row in cursor.fetchall())
    conn.close()
    
    return render_template("search_heart.html",
                         sex_types=sex_types,
                         target_types=target_types,
                         name=session.get('login_name'),
                         login_status=login_status,
                         user_recent_records=user_recent_records)

@app.post("/table_heart")
def table_heart():
    return render_template("table.html",data=query_heart(dict(request.form)))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    no_error=True
    if request.method == "POST":
        entered_name = request.form.get("name")
        entered_password = request.form.get("password")
        sex=request.form.get("sex")
        if request.form.get("age"):
            age=int(request.form.get("age"))
        else:
            age=''
        #user_df = pd.read_csv('user_data.csv')
        conn = sqlite3.connect("user_db.sqlite3")
        user_df = pd.read_sql_query("SELECT * FROM user", conn)

        #fracture_df=pd.read_csv('fracture_data.csv')
        if not (entered_name and entered_password):
            flash("Please enter all the required information.")
            no_error=False
        if  not user_df[user_df['name'] == entered_name].empty:
            flash("Username already exists. Please choose another.")
            no_error=False
        if no_error==False:
            return redirect(url_for("signup"))
        
        new_user = pd.DataFrame({'name': [entered_name], 'password': [entered_password]})
        new_user.to_sql('user', conn, if_exists='append', index=False)

        temp_name = [entered_name]
        temp_sex = [sex]
        temp_age = [age]   
        temp_df = pd.DataFrame({'name':temp_name,'sex':temp_sex,'age':temp_age,'fracture':['']})
        temp_df.to_sql('fracture', conn, if_exists='append', index=False)
        temp_df = pd.DataFrame({'name':temp_name,'sex':temp_sex,'age':temp_age,'target':['']})
        temp_df.to_sql('heart', conn, if_exists='append', index=False)
        conn.close()
        #temp_df.to_csv("fracture_data.csv",mode='a',index=False,header=False)
        #df=pd.read_csv('fracture_data.csv',na_values=fracture_id)
        # user_df = pd.concat([user_df, new_user], ignore_index=True)
        #print(new_user)
        #new_user.to_csv('user_data.csv', mode='a', header=False, index=False)
        # user_df.to_csv('user_data.csv', index=False)
        flash("Sign up successful! Please log in.")
        return redirect(url_for("login"))
    else:
        return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    signup_just_finished = True
    if request.method == "POST":
        print("=== Login request started ===")
        print(f"Request method: {request.method}")
        print(f"Form data: {request.form}")
        
        entered_name = request.form.get("name")
        entered_password = request.form.get("password")
        
        print(f"Entered username: {entered_name}")
        print(f"Entered password: {entered_password}")
        
        # user_df = pd.read_csv('user_data.csv')
        conn = sqlite3.connect("user_db.sqlite3")
        user_df = pd.read_sql_query("SELECT * FROM user", conn)
        name_index = user_df[user_df['name'] == entered_name]
        print(f"Database query result: {name_index}")
        print(f"Query result type: {type(name_index)}")
        print(f"Query result is empty: {name_index.empty}")
        conn.close()
        
        # print(user_df)
        # print(name_row)
        if name_index.empty and entered_name:
            print("User not found; redirecting to signup")
            flash("No user found. Please sign up.")
            signup_just_finished = False
            return redirect(url_for("signup"))
        if not (entered_name and entered_password):
            print("Username or password is empty")
            flash("Please enter all the required information.")
            signup_just_finished = False
            return redirect(url_for("login"))
        else:
            correct_password = name_index.iloc[0]['password']
            print(f"Password in database: {correct_password}")
            print(f"Password matches: {correct_password == entered_password}")
            
            if correct_password != entered_password:
                print("Incorrect password")
                flash("Wrong password.")
                signup_just_finished = False
                return redirect(url_for("login"))
            else:
                print("Login successful; setting the session")
                flash(f"Welcome, {entered_name}")
                session['login_name'] = entered_name
                print(f"Session after update: {session.get('login_name')}")
                return redirect(url_for("home"))
                
    return render_template("login.html", signup_just_finished=signup_just_finished)

@app.route("/logout")
def logout():
    session.pop('login_name', None)  # Safely remove the session value
    flash("You have been logged out successfully")
    return redirect(url_for("home"))

@app.route("/charts")
def charts():
    fracture_data_presence=True
    heart_data_presence=True

    conn = sqlite3.connect("user_db.sqlite3")
    raw_df = pd.read_sql_query("SELECT * FROM fracture", conn)
    df = raw_df[(raw_df['fracture'] != 'No Fracture') & (raw_df['fracture'] != '')]
    conn.close()
    print(df.head())
    print(df.info())
    print(df.shape)
    sex_vc=df['sex'].value_counts()
    print(sex_vc)
    sex_pie_values=[]
    sex_pie_labels=[]
    if 'male'in sex_vc:
        sex_pie_values.append(sex_vc['male'])
        sex_pie_labels.append("male")
    if 'female' in sex_vc:
        sex_pie_values.append(sex_vc['female'])
        sex_pie_labels.append("female")
    if not sex_pie_values:
        fracture_data_presence=False  
    else:
        plt.pie(sex_pie_values, labels=sex_pie_labels, autopct=make_autopct(sex_pie_values), startangle=90)
        plt.title('Sex Distribution (Fracture)')
        plt.axis('equal')
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Directory containing this Python file
        save_dir = os.path.join(current_dir, 'static')  # Static subdirectory
        save_path_fracture_sex = os.path.join(save_dir, 'sex_chart_fracture.jpg')  # Output file path
        plt.savefig(save_path_fracture_sex)
        plt.close()

    conn = sqlite3.connect("user_db.sqlite3")
    raw_df = pd.read_sql_query("SELECT * FROM heart", conn)
    df = raw_df[raw_df['target'] == 'yes']
    conn.close()
    print(df.head())
    print(df.info())
    print(df.shape)
    sex_vc=df['sex'].value_counts()
    print(sex_vc)
    sex_pie_values=[]
    sex_pie_labels=[]
    if 'male'in sex_vc:
        sex_pie_values.append(sex_vc['male'])
        sex_pie_labels.append("male")
    if 'female' in sex_vc:
        sex_pie_values.append(sex_vc['female'])
        sex_pie_labels.append("female")
    if not sex_pie_values:
        heart_data_presence=False
    else:    
        plt.pie(sex_pie_values, labels=sex_pie_labels, autopct=make_autopct(sex_pie_values), startangle=90)
        plt.title('Sex Distribution (Heart Disease)')
        plt.axis('equal')
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Directory containing this Python file
        save_dir = os.path.join(current_dir, 'static')  # Static subdirectory
        save_path_heart_sex = os.path.join(save_dir, 'sex_chart_heart.jpg')  # Output file path
        plt.savefig(save_path_heart_sex)
        plt.close()

    conn = sqlite3.connect("user_db.sqlite3")
    raw_df = pd.read_sql_query("SELECT * FROM fracture", conn)
    df = raw_df[(raw_df['fracture'] != 'No Fracture') & (raw_df['fracture'] != '')]
    conn.close()
    print(df.head())
    print(df.info())
    print(df.shape)
    if df.empty:
        fracture_data_presence=False
    else:
        bins = range(0, 121, 5)
        ax = sns.histplot(df['age'], bins=bins, color='skyblue')
        for p in ax.patches:
            height = p.get_height()
            if height!=0:
                ax.text(p.get_x() + p.get_width()/2, height + 0, int(height), ha='center', fontsize=10)
        plt.title('Age Distribution (Fracture)')
        plt.xlabel('Age')
        plt.ylabel('Number of People')
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Directory containing this Python file
        save_dir = os.path.join(current_dir, 'static')  # Static subdirectory
        save_path_fracture_age = os.path.join(save_dir, 'age_chart_fracture.jpg')  # Output file path
        plt.savefig(save_path_fracture_age)
        plt.close()

    conn = sqlite3.connect("user_db.sqlite3")
    raw_df = pd.read_sql_query("SELECT * FROM heart", conn)
    df = raw_df[raw_df['target'] == 'yes']
    conn.close()
    print(df.head())
    print(df.info())
    print(df.shape)
    if df.empty:
        heart_data_presence=False
    else:
        bins = range(0, 121, 5)
        ax = sns.histplot(df['age'], bins=bins, color='skyblue')
        for p in ax.patches:
            height = p.get_height()
            if height!=0:
                ax.text(p.get_x() + p.get_width()/2, height + 0, int(height), ha='center', fontsize=10)
        plt.title('Age Distribution (Heart Disease)')
        plt.xlabel('Age')
        plt.ylabel('Number of People')
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Directory containing this Python file
        save_dir = os.path.join(current_dir, 'static')  # Static subdirectory
        save_path_heart_age = os.path.join(save_dir, 'age_chart_heart.jpg')  # Output file path
        plt.savefig(save_path_heart_age)
        plt.close()

    conn = sqlite3.connect("user_db.sqlite3")
    raw_df = pd.read_sql_query("SELECT * FROM fracture", conn)
    df = raw_df[(raw_df['fracture'] != 'No Fracture') & (raw_df['fracture'] != '')]
    conn.close()
    print(df.head())
    print(df.info())
    print(df.shape)
    if df.empty:
        fracture_data_presence=False
    else:    
        counts = df['fracture'].value_counts()
        bars = plt.bar(counts.index, counts.values, color='skyblue')  # Draw the bar chart

        # Add a count label to each bar
        for bar in bars:
            height = bar.get_height()  # Bar height
            if height!=0:
                plt.text(bar.get_x() + bar.get_width()/2, height + 0, int(height), ha='center', fontsize=10)  # Show the integer count, centered horizontally

        plt.xlabel('Fracture Category')
        plt.ylabel('Number of People')
        plt.title('Fracture Category Distribution')
        # plt.xticks(rotation=45)  # Rotate long category labels to prevent overlap
        plt.tight_layout()  # Adjust the layout to prevent clipped labels
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Directory containing this Python file
        save_dir = os.path.join(current_dir, 'static')  # Static subdirectory
        save_path_fracture_classification = os.path.join(save_dir, 'classification_chart_fracture.jpg')  # Output file path
        plt.savefig(save_path_fracture_classification)
        plt.close()

    print(f"fracture_data_presence: {fracture_data_presence}\nheart_data_presence: {heart_data_presence}")
    return render_template("charts.html",fracture_data_presence=fracture_data_presence,heart_data_presence=heart_data_presence)
    

@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404

if __name__ == '__main__':
    # Production configuration
    app.run(
        host='0.0.0.0',  # Bind to all network interfaces to allow external access
        port=80,        # Set the port number
        debug=False,      # Disable debug mode for security
        threaded=True     # Enable multithreading
    )
    
