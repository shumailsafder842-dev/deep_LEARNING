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


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "FAST_FOOD_CLASSIFICATION.keras")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (128, 128)          # must match training size

# Must match the class order used during training.
# NOTE: if you trained with flow_from_directory / image_dataset_from_directory,
# classes are sorted ALPHABETICALLY -> ["Donut", "Pizza", "Taco"].
# If your predictions come out mislabeled, change the order below.
CLASS_NAMES = ["Donut", "Pizza", "Taco"]

# Set to False if your model already contains a Rescaling(1./255) layer
# or you did NOT divide images by 255 during training.
RESCALE_255 = True

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


# ==================================================
# 2. FLASK APPLICATION
# ==================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "change-this-secret-key-before-deployment"
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# ==================================================
# 3. LOAD TRAINED MODEL
# ==================================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)

# Sanity check: compare configured size with what the model expects
try:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    expected = (shape[1], shape[2])
    if None not in expected and tuple(expected) != tuple(IMG_SIZE):
        print(f"WARNING: model expects {expected} but IMG_SIZE is {IMG_SIZE}")
except Exception:
    pass

print("Fast Food model loaded successfully!")
print(f"Model path : {MODEL_PATH}")
print(f"Input shape: {model.input_shape}")


# ==================================================
# 4. DEMO USERS  (admin/admin123, user/user123)
# ==================================================

USERS = {
    "admin": generate_password_hash("admin123"),
    "user": generate_password_hash("user123"),
}


# ==================================================
# 5. HELPER FUNCTIONS
# ==================================================

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
    """Load, preprocess and classify an uploaded image."""
    img = keras_image.load_img(image_path, target_size=IMG_SIZE, color_mode="rgb")
    img_array = keras_image.img_to_array(img)

    if RESCALE_255:
        img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)      # (1, 128, 128, 3)

    predictions = model.predict(img_array, verbose=0)
    probabilities = np.asarray(predictions[0]).reshape(-1)

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {len(probabilities)} outputs, "
            f"but {len(CLASS_NAMES)} class names are configured."
        )

    idx = int(np.argmax(probabilities))
    label = CLASS_NAMES[idx]
    confidence = round(float(probabilities[idx]) * 100, 2)
    all_probs = {
        CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }
    return label, confidence, all_probs


# ==================================================
# 6. HOME
# ==================================================

@app.route("/")
def home():
    return render_template("home.html")


# ==================================================
# 7. LOGIN
# ==================================================

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
            if (
                next_page.startswith("/")
                and not next_page.startswith("//")
                and "\\" not in next_page
            ):
                return redirect(next_page)
            return redirect(url_for("prediction"))

        flash("Invalid username or password.", "danger")

    return render_template("login.html")


# ==================================================
# 8. LOGOUT
# ==================================================

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


# ==================================================
# 9. PREDICTION (login required)
# ==================================================

@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    result = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if uploaded_file is None or uploaded_file.filename == "":
            flash("Please select an image first.", "danger")

        elif not allowed_file(uploaded_file.filename):
            flash("Only PNG, JPG, JPEG and WEBP images are allowed.", "danger")

        else:
            filename = f"{uuid.uuid4().hex}_{secure_filename(uploaded_file.filename)}"
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            try:
                uploaded_file.save(image_path)
                label, confidence, probabilities = predict_image(image_path)

                result = {
                    "label": label,
                    "confidence": confidence,
                    "probs": probabilities,
                    "image": url_for("static", filename=f"uploads/{filename}"),
                }

            except Exception as e:
                # Full traceback prints in the terminal
                app.logger.exception("Failed to process uploaded image.")
                flash(f"Could not process this image: {e}", "danger")

    return render_template("prediction.html", result=result, user=session["user"])


# ==================================================
# 10. FILE TOO LARGE
# ==================================================

@app.errorhandler(413)
def file_too_large(error):
    flash("Image is too large. Maximum file size is 5 MB.", "danger")
    return redirect(url_for("prediction"))


# ==================================================
# 11. RUN
# ==================================================

if __name__ == "__main__":
    # use_reloader=False stops Flask from loading the model twice in debug mode
    app.run(debug=True, use_reloader=False)