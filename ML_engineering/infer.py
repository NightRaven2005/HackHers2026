import torch
import torch.nn as nn
import joblib
import pandas as pd
import random
import glob
import librosa
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

scaler = joblib.load('encoders/scaler.pkl')
le = joblib.load('encoders/label_encoder.pkl')

INPUT_SIZE = 47
NUM_CLASSES = 10

model = MLP(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('encoders/model_3sec.pth', weights_only=True))
model.eval()

def _extract_features(audio_path: str) -> dict:
    y_audio, sr = librosa.load(audio_path, duration=3.0)
    features = {
        'rms_mean': float(np.mean(librosa.feature.rms(y=y_audio))),
        'rms_var': float(np.var(librosa.feature.rms(y=y_audio))),
        'harmony_mean': float(np.mean(librosa.effects.harmonic(y=y_audio))),
        'harmony_var': float(np.var(librosa.effects.harmonic(y=y_audio))),
        'perceptr_mean': float(np.mean(librosa.effects.percussive(y=y_audio))),
        'perceptr_var': float(np.var(librosa.effects.percussive(y=y_audio))),
        'tempo': float(librosa.beat.tempo(y=y_audio, sr=sr)[0]),
    }
    mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = float(np.mean(mfccs[i-1]))
        features[f'mfcc{i}_var'] = float(np.var(mfccs[i-1]))
    return features

def predict_genre(audio_path: str) -> dict:
    raw_features = _extract_features(audio_path)
    features_df = pd.DataFrame([raw_features], columns=scaler.feature_names_in_)
    features_scaled = scaler.transform(features_df)
    with torch.no_grad():
        inputs = torch.FloatTensor(features_scaled)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs / 1.5, dim=1).numpy()[0]
    genre = le.inverse_transform(predicted.numpy())[0]
    probs = {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(probabilities)}
    return {"genre": genre, "probabilities": probs}

if __name__ == "__main__":
    df = pd.read_csv('datasets/Data/features_3_sec_cleaned.csv')
    sample = df.iloc[0]
    actual_label = le.inverse_transform([int(sample["label"])])[0]
    features_scaled = sample.drop(["label", "filename"]).values.astype("float32")

    with torch.no_grad():
        inputs = torch.FloatTensor(features_scaled).unsqueeze(0)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    predicted_label = le.inverse_transform(predicted.numpy())[0]

    print(f"Actual genre:    {actual_label}")
    print(f"Predicted genre: {predicted_label}")

    files = glob.glob('datasets/Data/genres_original/**/*.wav', recursive=True)
    sampled_files = random.sample(files, 10)
    print("\nRandom sample predictions:")
    for file in sampled_files:
        result = predict_genre(file)
        actual_genre = file.split('\\')[-2]
        print(f"  {actual_genre:10s} -> {result['genre']}")
        for genre, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {genre:12s}: {prob:.1%}")