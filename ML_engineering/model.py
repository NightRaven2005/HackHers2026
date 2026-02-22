import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df1 = pd.read_csv('datasets/Data/features_3_sec_cleaned.csv')
X = df1.drop(['label', 'filename'], axis=1).values
y = df1['label'].values
groups = df1['filename']
gss = GroupShuffleSplit(test_size=0.2,  random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups))
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

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

input_size = X_train.shape[1]
num_classes = len(np.unique(y))
model = MLP(input_size=input_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

print("Starting training...")
for epoch in range(100):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            val_loss += criterion(outputs, y_batch).item()

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}')

print("\nEvaluating on validation set...")
model.eval()
all_preds = []
with torch.no_grad():
    for X_batch, _ in val_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())

print("Validation Accuracy:", accuracy_score(y_val, all_preds))
print("Classification Report:\n", classification_report(y_val, all_preds))
print("Confusion Matrix:\n", confusion_matrix(y_val, all_preds))

torch.save(model.state_dict(), 'model_3sec.pth', _use_new_zipfile_serialization=False)
print("Model saved to model_3sec.pth")
