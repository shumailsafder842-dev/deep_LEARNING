import os
import uuid
from functools import wraps

import numpy as np
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "Intel_Scene_Classification.keras")  # <-- your saved model

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (150, 150)

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

RESCALE_255 = True

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "change-this-secret-key-before-deployment"
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE



if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)

try:
    _shape = model.input_shape
    if isinstance(_shape, list):
        _shape = _shape[0]
    if _shape[1] and _shape[2]:
        IMG_SIZE = (int(_shape[1]), int(_shape[2]))
except Exception:
    pass

print("Scene model loaded successfully!")
print(f"Input shape: {model.input_shape} | using IMG_SIZE = {IMG_SIZE}")



USERS = {
    "admin": generate_password_hash("admin123"),
    "user": generate_password_hash("user123"),
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to use the prediction page.", "warning")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def predict_image(image_path):
    img = keras_image.load_img(image_path, target_size=IMG_SIZE, color_mode="rgb")
    arr = keras_image.img_to_array(img)
    if RESCALE_255:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)
    probs = np.asarray(preds[0]).reshape(-1)

    if len(probs) != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {len(probs)} outputs, "
            f"but {len(CLASS_NAMES)} class names are configured."
        )

    idx = int(np.argmax(probs))
    all_probs = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
                 for i in range(len(CLASS_NAMES))}
    # sort high -> low for display
    all_probs = dict(sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True))
    return CLASS_NAMES[idx], round(float(probs[idx]) * 100, 2), all_probs



@app.route("/")
def home():
    return render_template("home.html", classes=CLASS_NAMES)


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("prediction"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if username in USERS and check_password_hash(USERS[username], password):
            session["user"] = username
            flash(f"Welcome back, {username}!", "success")

            next_page = request.args.get("next", "")
            if (next_page.startswith("/") and not next_page.startswith("//")
                    and "\\" not in next_page):
                return redirect(next_page)
            return redirect(url_for("prediction"))

        flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    result = None

    if request.method == "POST":
        f = request.files.get("image")

        if f is None or f.filename == "":
            flash("Please select an image first.", "danger")
        elif not allowed_file(f.filename):
            flash("Only PNG, JPG, JPEG and WEBP images are allowed.", "danger")
        else:
            filename = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            try:
                f.save(path)
                label, confidence, probs = predict_image(path)
                result = {
                    "label": label,
                    "confidence": confidence,
                    "probs": probs,
                    "image": url_for("static", filename=f"uploads/{filename}"),
                }
            except Exception as e:
                app.logger.exception("Failed to process uploaded image.")
                flash(f"Could not process this image: {e}", "danger")

    return render_template("prediction.html", result=result, user=session["user"])


@app.errorhandler(413)
def file_too_large(error):
    flash("Image is too large. Maximum file size is 5 MB.", "danger")
    return redirect(url_for("prediction"))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)