import os
import io
import base64
import ast
import operator as op
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
from flask import Flask, render_template, request, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras import models

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "trained_model.h5"   # change if needed
DEFAULT_THRESHOLD = 0.70
APPLY_SMOOTHING = True            # server-side smoothing (Gaussian blur) to help handwriting
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "ADGHGH234567DFG"  # <<-- replace with secure random value for production

# -----------------------------
# Load model (prediction-only)
# -----------------------------
model = None
model_load_error = None

def try_load_model(path):
    global model, model_load_error
    if not os.path.exists(path):
        model_load_error = f"Model file not found at '{path}'. Please place a trained model there."
        print("[WARN]", model_load_error)
        model = None
        return

    try:
        model = models.load_model(path, compile=False)
        model_load_error = None
        print(f"[INFO] Loaded model from {path}")
    except Exception as e:
        model = None
        model_load_error = f"Failed to load model: {e}"
        print("[ERROR]", model_load_error)

try_load_model(MODEL_PATH)


# -----------------------------
# Preprocessing utilities
# -----------------------------
def center_and_resize_to_28(img_L_inverted: Image.Image) -> Image.Image:
    arr = np.array(img_L_inverted)
    rows = np.where(arr.max(axis=1) > 0)[0]
    cols = np.where(arr.max(axis=0) > 0)[0]

    if rows.size and cols.size:
        r0, r1 = rows[0], rows[-1] + 1
        c0, c1 = cols[0], cols[-1] + 1
        arr = arr[r0:r1, c0:c1]

    if arr.size == 0:
        return Image.new('L', (28, 28), 0)

    h, w = arr.shape
    if h > w:
        new_h, new_w = 20, max(1, int(round(20 * w / h)))
    else:
        new_w, new_h = 20, max(1, int(round(20 * h / w)))

    digit = Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new('L', (28, 28), 0)
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    canvas.paste(digit, (left, top))
    return canvas

def preprocess_pil_to_tensor(img: Image.Image, apply_smoothing: bool = True) -> np.ndarray:
    """
    Convert PIL image to MNIST-like tensor:
      - grayscale
      - optional smoothing (Gaussian blur) to reduce stroke noise
      - invert (white digit on black)
      - center+resize to 28x28
      - normalize and reshape to (1,28,28,1)
    """
    img = img.convert("L")
    if apply_smoothing:
        # small blur helps thin strokes and reduces jitter
        img = img.filter(ImageFilter.GaussianBlur(radius=0.9))
    img = ImageOps.invert(img)
    img = center_and_resize_to_28(img)
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

def predict_with_confidence_pil(img: Image.Image, threshold: float, smoothing: bool):
    if model is None:
        raise RuntimeError(model_load_error or "Model not loaded.")
    tensor = preprocess_pil_to_tensor(img, apply_smoothing=smoothing)
    probs = model.predict(tensor, verbose=0)[0]
    label = int(np.argmax(probs))
    conf = float(np.max(probs))
    is_sure = conf >= threshold
    return label, conf, is_sure, probs.tolist()


# -----------------------------
# Safe eval (AST-based)
# -----------------------------
# Allowed operators and mapping
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

def safe_eval(node):
    """
    Evaluate an AST node safely supporting numbers, binary ops and parentheses.
    """
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    if isinstance(node, ast.Num):  # Python <3.8
        return node.n
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant):  # Python 3.8+
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Unsupported constant type")
    if isinstance(node, ast.BinOp):
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported operator {op_type}")
    if isinstance(node, ast.UnaryOp):
        operand = safe_eval(node.operand)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator {op_type}")
    raise ValueError(f"Unsupported expression: {type(node)}")

def evaluate_expression_safely(expr: str):
    """
    Evaluate expression string after minor normalization:
      - Replace unicode multiplication/division signs if present
      - Ensure only allowed characters appear (digits, operators, parentheses, dot, spaces)
    """
    if expr is None:
        raise ValueError("Empty expression")
    # normalize display operators to python ones
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    allowed_chars = "0123456789.+-*/()% "
    # quick check: after replacing '**' keep allowed
    tmp = expr.replace("**", "")  # temporarily remove power token for check
    if not all(ch in allowed_chars for ch in tmp):
        raise ValueError("Invalid characters in expression")
    # parse and evaluate using ast
    parsed = ast.parse(expr, mode='eval')
    return safe_eval(parsed)


# -----------------------------
# Digit segmentation (for /predict)
# -----------------------------
def segment_digits(image):
    """
    Takes a binary canvas image and segments into individual digits.
    Returns list of digit images (28x28).
    """
    # Convert to grayscale - support RGBA/BGR etc
    if image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours (each digit)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_imgs = []
    boxes = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > 100:  # ignore tiny noise
            boxes.append((x, y, w, h))
    
    # Sort left-to-right
    boxes = sorted(boxes, key=lambda b: b[0])
    
    for (x, y, w, h) in boxes:
        digit = thresh[y:y+h, x:x+w]
        # Pad to square
        pad = abs(h - w) // 2
        if h > w:
            digit = cv2.copyMakeBorder(digit, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
        else:
            digit = cv2.copyMakeBorder(digit, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
        
        # Resize to MNIST format
        digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        digit = digit.astype("float32") / 255.0
        digit = np.expand_dims(digit, axis=(0, -1))  # (1, 28, 28, 1)
        
        digit_imgs.append(digit)
    
    return digit_imgs


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model_status", methods=["GET"])
def model_status():
    if model is None:
        return jsonify({"loaded": False, "error": model_load_error})
    return jsonify({"loaded": True, "model_path": MODEL_PATH})


@app.route("/predict_draw", methods=["POST"])
def predict_draw():
    """
    Body JSON: { image: "data:image/png;base64,...", threshold: 0.7?, smoothing: true|false? }
    """
    try:
        if model is None:
            return jsonify({"error": model_load_error or "Model is not loaded."}), 400

        data = request.get_json(force=True)
        data_url = data.get("image")
        threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
        smoothing = bool(data.get("smoothing", APPLY_SMOOTHING))

        header, b64data = data_url.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        img = Image.open(io.BytesIO(img_bytes))

        label, conf, is_sure, probs = predict_with_confidence_pil(img, threshold, smoothing)
        return jsonify({
            "digit": None if not is_sure else label,
            "raw_digit": label,
            "confidence": conf,
            "is_sure": is_sure,
            "probs": probs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict_files", methods=["POST"])
def predict_files():
    try:
        if model is None:
            return jsonify({"error": model_load_error or "Model not loaded."}), 400

        threshold = float(request.form.get("threshold", DEFAULT_THRESHOLD))
        smoothing = request.form.get("smoothing", str(APPLY_SMOOTHING)).lower() in ("1","true","yes")
        files = request.files.getlist("files")
        results = []
        for f in files:
            img = Image.open(f.stream)
            label, conf, is_sure, probs = predict_with_confidence_pil(img, threshold, smoothing)
            results.append({
                "filename": f.filename,
                "digit": None if not is_sure else label,
                "raw_digit": label,
                "confidence": conf,
                "is_sure": is_sure,
                "probs": probs
            })
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# New: predict + append (atomic)
@app.route("/predict_and_append", methods=["POST"])
def predict_and_append():
    """
    Accepts same body as predict_draw; if digit predicted confidently, append to session expression.
    Returns the prediction and updated expression.
    """
    try:
        if model is None:
            return jsonify({"error": model_load_error or "Model not loaded."}), 400

        data = request.get_json(force=True)
        data_url = data.get("image")
        threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
        smoothing = bool(data.get("smoothing", APPLY_SMOOTHING))

        header, b64data = data_url.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        img = Image.open(io.BytesIO(img_bytes))

        label, conf, is_sure, probs = predict_with_confidence_pil(img, threshold, smoothing)

        # ensure session key exists
        expr = session.get("expression", "")
        appended = False
        if is_sure:
            expr = expr + str(label)
            session["expression"] = expr
            appended = True

        return jsonify({
            "digit": None if not is_sure else label,
            "raw_digit": label,
            "confidence": conf,
            "is_sure": is_sure,
            "probs": probs,
            "appended": appended,
            "expression": expr
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Append symbol (operators or digits) via API (centralized)
@app.route("/append_symbol", methods=["POST"])
def append_symbol():
    try:
        data = request.get_json(force=True)
        symbol = data.get("symbol", "")
        if not isinstance(symbol, str) or not symbol:
            return jsonify({"error": "Missing symbol"}), 400

        # normalize common display operators
        symbol = symbol.replace("×", "*").replace("÷", "/").replace("^", "**")

        # one more validation: only allow small set of tokens
        allowed = set("0123456789+-*/()%. ")
        test = symbol.replace("**", "")  # remove power token before checking allowed chars
        if not all(ch in allowed for ch in test):
            return jsonify({"error": "Invalid symbol"}), 400

        expr = session.get("expression", "")
        expr = expr + symbol
        session["expression"] = expr
        return jsonify({"expression": expr})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/clear_expression", methods=["POST"])
def clear_expression():
    try:
        session["expression"] = ""
        return jsonify({"expression": ""})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/get_expression", methods=["GET"])
def get_expression():
    try:
        expr = session.get("expression", "")
        return jsonify({"expression": expr})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/evaluate_expression", methods=["POST"])
def evaluate_expression():
    try:
        # Check if an expression was supplied in the body, otherwise use session one
        data = request.get_json(force=True) if request.data else {}
        expr = data.get("expression", None)
        if expr is None:
            expr = session.get("expression", "")

        # evaluate safely
        result = evaluate_expression_safely(expr)
        # optionally clear expression or keep it — we keep it (client can clear explicitly)
        return jsonify({"expression": expr, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    
    # Segment digits
    digits = segment_digits(img)
    preds = []
    
    for d in digits:
        prediction = model.predict(d)
        preds.append(str(np.argmax(prediction)))
    
    result = "".join(preds)  # e.g., "123"
    
    return jsonify({"prediction": result})


# -----------------------------
# File management (uploads)
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Accepts multipart form uploads.
    Returns: { files: [ { filename, size }, ... ] }
    """
    if not request.files:
        return jsonify({"error": "No files uploaded"}), 400

    saved = []
    # accept any file field
    for key in request.files:
        for f in request.files.getlist(key):
            if f and f.filename:
                filename = secure_filename(f.filename)
                # make unique if exists
                dest = os.path.join(UPLOAD_DIR, filename)
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest):
                    filename = f"{base}_{counter}{ext}"
                    dest = os.path.join(UPLOAD_DIR, filename)
                    counter += 1
                f.save(dest)
                saved.append({"filename": filename, "size": os.path.getsize(dest)})
    return jsonify({"files": saved})


@app.route("/list_uploads", methods=["GET"])
def list_uploads():
    files = []
    for fname in sorted(os.listdir(UPLOAD_DIR), reverse=True):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if os.path.isfile(fpath):
            files.append({"filename": fname, "size": os.path.getsize(fpath)})
    return jsonify({"files": files})


@app.route("/delete_file/<path:filename>", methods=["DELETE"])
def delete_file(filename):
    # secure the filename to avoid traversal attacks
    filename = secure_filename(filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath) and os.path.isfile(filepath):
        os.remove(filepath)
        return jsonify({"success": True})
    return jsonify({"error": "File not found"}), 404


@app.route("/uploads/<path:filename>")
def serve_file(filename):
    # security: send only from UPLOAD_DIR
    return send_from_directory(UPLOAD_DIR, filename)


# -----------------------------
if __name__ == "__main__":
    print("[INFO] Starting Flask app (prediction-only).")
    app.run(debug=True)
