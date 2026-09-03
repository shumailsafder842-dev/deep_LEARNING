from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pickle
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)

app.secret_key = "imdb_rnn_secret_key"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "IMDB_Simple_RNN.keras"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "imdb_tokenizer.pkl"
)

print("Model path:", MODEL_PATH)
print("Tokenizer path:", TOKENIZER_PATH)

print("Model exists:", os.path.exists(MODEL_PATH))
print("Tokenizer exists:", os.path.exists(TOKENIZER_PATH))


model = load_model(MODEL_PATH)


with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")

        # Basic validation
        if not name or not email or not password:
            return render_template(
                "login.html",
                error="Please fill all fields."
            )

        session["logged_in"] = True
        session["name"] = name
        session["email"] = email

        return redirect(url_for("prediction"))

    return render_template("login.html")



@app.route("/prediction", methods=["GET", "POST"])
def prediction():


    if not session.get("logged_in"):
        return redirect(url_for("login"))

    result = None
    probability = None
    review = ""

    if request.method == "POST":

        review = request.form.get("review", "").strip()

        if not review:
            return render_template(
                "prediction.html",
                name=session.get("name"),
                error="Please enter an IMDb review."
            )

        sequence = tokenizer.texts_to_sequences([review])

        padded_review = pad_sequences(
            sequence,
            maxlen=200,
            padding="post",
            truncating="post"
        )

        prediction_value = model.predict(
            padded_review,
            verbose=0
        )[0][0]

        probability = float(prediction_value) * 100

        if prediction_value >= 0.5:
            result = "Positive"
            confidence = probability
        else:
            result = "Negative"
            confidence = 100 - probability

        return render_template(
            "prediction.html",
            name=session.get("name"),
            review=review,
            result=result,
            probability=round(confidence, 2)
        )

    return render_template(
        "prediction.html",
        name=session.get("name")
    )



@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("home"))



if __name__ == "__main__":
    app.run(debug=True)