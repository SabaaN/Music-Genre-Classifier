from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
import librosa
import os
import math
from pydub import AudioSegment
import subprocess

app = Flask(__name__)

# Load model once at startup
model = load_model("./models/model_cnn3.h5")

# Define genre dictionary
genre_dict = {
    0: "disco", 1: "pop", 2: "classical", 3: "metal", 4: "rock",
    5: "blues", 6: "hiphop", 7: "reggae", 8: "country", 9: "jazz"
}

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def homepage():
    return render_template('homepage.html', title="MGC")

@app.route("/prediction", methods=["POST"])
def prediction():
    title = "MGC | Prediction"

    if 'myfile' not in request.files:
        return "No file uploaded.", 400

    audio_file = request.files['myfile']
    filename = secure_filename(audio_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(filepath)

    # Convert to .wav if it's an MP3
    if filename.endswith(".mp3"):
        wav_path = filepath.rsplit('.', 1)[0] + '.wav'
        subprocess.call(['ffmpeg', '-y', '-i', filepath, wav_path])
        filepath = wav_path

    # Trim the audio: 60s to 90s
    segment = AudioSegment.from_wav(filepath)
    trimmed = segment[60 * 1000:90 * 1000]  # milliseconds
    trimmed.export(filepath, format="wav")

    # Process input
    def process_input(audio_file, track_duration):
        SAMPLE_RATE = 22050
        NUM_MFCC = 13
        N_FTT = 2048
        HOP_LENGTH = 512
        SAMPLES_PER_TRACK = SAMPLE_RATE * track_duration
        NUM_SEGMENTS = 10

        samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
        signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)

        # Get MFCCs from first segment only
        mfcc = librosa.feature.mfcc(signal[:samples_per_segment], sample_rate, n_mfcc=NUM_MFCC,
                                    n_fft=N_FTT, hop_length=HOP_LENGTH)
        return mfcc.T

    mfcc = process_input(filepath, 30)
    X_to_predict = mfcc[np.newaxis, ..., np.newaxis]

    # Make prediction
    pred = model.predict(X_to_predict)
    pred_class = np.argmax(pred)

    probabilities = pred[0]
    top_indices = np.argsort(probabilities)[-3:][::-1]

    return render_template(
        'prediction.html',
        title=title,
        prediction=genre_dict[top_indices[0]],
        probability="{:.2f}".format(probabilities[top_indices[0]] * 100),
        second_prediction=genre_dict[top_indices[1]],
        second_probability="{:.2f}".format(probabilities[top_indices[1]] * 100),
        third_prediction=genre_dict[top_indices[2]],
        third_probability="{:.2f}".format(probabilities[top_indices[2]] * 100),
    )

if __name__ == "__main__":
    app.run(debug=True)
