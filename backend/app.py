from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from services.generator import generate_from_blend
from services.classifier import predict_genre

app = Flask(__name__, static_folder="static")
CORS(app)

AUDIO_DIR = os.path.join(app.static_folder, "audio")
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def health():
    return jsonify({"status": "Flask backend running"})

@app.post("/blend")
def blend():
    data = request.get_json(force=True)
    genreA = data["genreA"]
    genreB = data["genreB"]
    alpha = float(data["alpha"])

    audio_path = generate_from_blend(genreA, genreB, alpha, out_dir=AUDIO_DIR)
    audio_url = f"/static/audio/{os.path.basename(audio_path)}"
    preds = predict_genre(audio_path)

    return jsonify({"audio_url": audio_url, "predictions": preds})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)