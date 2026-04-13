# 10-Fold Version of Two-Step Training
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random

# Set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
df = pd.read_csv("D:/WESAD_output/subjectwise_zscore_normalize60s.csv")
X_all = df.iloc[:, 2:2562].values
y_gender = df['Gender'].values

df = df[df['Label'].isin(['baseline', 'amusement', 'stress'])]
df['EmotionBinary'] = df['Label'].map(lambda x: 0 if x in ['baseline', 'amusement'] else 1)
y_emotion = df['EmotionBinary'].values

# Dataset
class ECGDataset(Dataset):
    def __init__(self, X, y, task='gender'):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y = np.array(y).astype(np.int64 if task=='emotion' else np.float32)
        self.y = torch.tensor(y, dtype=torch.long if task=='emotion' else torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# Models
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, 32, padding=15), nn.ReLU(),
            nn.MaxPool1d(8, 2),
            nn.Conv1d(32, 64, 16, padding=7), nn.ReLU(),
            nn.MaxPool1d(8, 2),
            nn.Conv1d(64, 128, 8, padding=3), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1))
    def forward(self, x): return self.feature_extractor(x).view(x.size(0), -1)

class GenderClassifier(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.classifier = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))
    def forward(self, x): return self.classifier(self.base(x))

class EmotionClassifier(nn.Module):
    def __init__(self, base, num_classes=2):
        super().__init__()
        self.base = base
        for param in self.base.parameters(): param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, num_classes))
    def forward(self, x): return self.classifier(self.base(x))

# Create output directory
os.makedirs("saved_models/twostep_10fold_2class", exist_ok=True)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
all_preds, all_labels = [], []
all_preds_gender, all_labels_gender = [], []
fold_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
    print(f"\n===== Fold {fold+1} =====")

    # === Gender Classifier Training ===
    base_model_gender = BaseModel().to(device)
    gender_model = GenderClassifier(base_model_gender).to(device)
    optimizer = optim.Adam(gender_model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    train_gen = ECGDataset(X_all[train_idx], y_gender[train_idx])
    test_gen = ECGDataset(X_all[test_idx], y_gender[test_idx])
    loader_gen = DataLoader(train_gen, batch_size=128, shuffle=True)

    best_acc = 0
    best_gender_state = None
    gender_model.train()
    for epoch in range(100):
        correct = 0
        for ecg, labels in loader_gen:
            ecg, labels = ecg.to(device), labels.to(device).view(-1,1)
            optimizer.zero_grad()
            outputs = gender_model(ecg)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = (torch.sigmoid(outputs) > 0.5)
            correct += (preds == labels).sum().item()
        acc = 100 * correct / len(train_gen)
        if acc > best_acc:
            best_acc = acc
            best_gender_state = base_model_gender.state_dict()

    torch.save(best_gender_state, f"saved_models/twostep_10fold_2class/base_gender_fold{fold+1}.pth")

    # === Gender Classifier Test ===
    gender_model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for ecg, label in DataLoader(test_gen):
            ecg = ecg.to(device)
            output = gender_model(ecg)
            pred = (torch.sigmoid(output) > 0.5).cpu().numpy().astype(int).flatten()
            preds.extend(pred)
            labels.extend(label.numpy().astype(int))
    all_preds_gender.extend(preds)
    all_labels_gender.extend(labels)

    # === Emotion Classifier Training ===
    base_model_emotion = BaseModel().to(device)
    base_model_emotion.load_state_dict(torch.load(f"saved_models/twostep_10fold_2class/base_gender_fold{fold+1}.pth", weights_only=True))
    emotion_model = EmotionClassifier(base_model_emotion).to(device)

    optimizer = optim.Adam(emotion_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_emo = ECGDataset(X_all[train_idx], y_emotion[train_idx], task='emotion')
    test_emo = ECGDataset(X_all[test_idx], y_emotion[test_idx], task='emotion')
    loader_emo = DataLoader(train_emo, batch_size=128, shuffle=True)

    best_acc = 0
    best_emotion_state = None
    emotion_model.train()
    for epoch in range(600):
        correct = 0
        for ecg, labels in loader_emo:
            ecg, labels = ecg.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = emotion_model(ecg)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
        acc = 100 * correct / len(train_emo)
        if acc > best_acc:
            best_acc = acc
            best_emotion_state = emotion_model.state_dict()

    torch.save(best_emotion_state, f"saved_models/twostep_10fold_2class/emotion_fold{fold+1}.pth")
    print(f"Saved twostep_10fold_2class/emotion_fold{fold+1}.pth (Best Train Acc: {best_acc:.2f}%)")

    # === Emotion Classifier Test ===
    emotion_model.load_state_dict(best_emotion_state)
    emotion_model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for ecg, label in DataLoader(test_emo):
            ecg = ecg.to(device)
            pred = emotion_model(ecg).argmax(1).cpu().numpy()
            preds.extend(pred)
            labels.extend(label.numpy())
    all_preds.extend(preds)
    all_labels.extend(labels)
    acc = 100 * np.mean(np.array(preds) == np.array(labels))
    fold_results.append((fold+1, acc))

# Final report
print("\n=== Gender Classification (10-Fold) ===")
print("Accuracy:", 100 * np.mean(np.array(all_preds_gender) == np.array(all_labels_gender)))
print("Confusion Matrix:\n", confusion_matrix(all_labels_gender, all_preds_gender))
print(classification_report(all_labels_gender, all_preds_gender, target_names=['Female', 'Male']))

print("\n=== Emotion Classification (10-Fold) ===")
print("Accuracy:", 100 * np.mean(np.array(all_preds) == np.array(all_labels)))
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=['Baseline/Amusement', 'Stress']))

print("\n=== 每 Fold 準確率 ===")
for fold_id, acc in fold_results:
    print(f"Fold {fold_id}: {acc:.2f}%")
