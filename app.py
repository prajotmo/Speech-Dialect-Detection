from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import librosa

app = Flask(__name__)

# Load the trained model
model = load_model("voice_classification_model.h5")

def extract_mfcc_features(audio_data, sample_rate, n_mfcc=15):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    return mfccs

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classify', methods=['POST'])
def classify_voice():
    # Assume the request contains a WAV file
    file = request.files['file']
    data, sample_rate = librosa.load(file, sr=None)  # Use librosa to load the audio file
    mfccs = extract_mfcc_features(data, sample_rate)
    # Pad or truncate the features to match the model's input shape
    max_length = 275  # Assuming the model's input shape is (275, 15)
    padded_mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])))
    features_flat = padded_mfccs.reshape(1, padded_mfccs.shape[0], padded_mfccs.shape[1])
    prediction = model.predict(features_flat)
    # Convert the prediction to a human-readable label
    result = "Marathwada Voice" if prediction < 1 else "Vidharbha Voice"
    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
