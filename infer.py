import torch
import torch.nn as nn
import joblib
import pandas as pd

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

scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

INPUT_SIZE = 47
NUM_CLASSES = 10

model = MLP(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('model_3sec.pth', weights_only=True))
model.eval()

def predict_genre(raw_features):
    feature_names = scaler.feature_names_in_
    features = pd.DataFrame([raw_features], columns=feature_names)
    features_scaled = scaler.transform(features)
    with torch.no_grad():
        inputs = torch.FloatTensor(features_scaled)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    return le.inverse_transform(predicted.numpy())[0]

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
