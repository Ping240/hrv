# === Fusion Model with Feature Concatenation (OneStep + TwoStep + HRV) ===

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import random
import neurokit2 as nk
from sklearn.model_selection import KFold

# ============ 固定隨機種子 ============
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 資料讀取與處理 ============
df = pd.read_csv("D:/WESAD_output/subjectwise_zscore_normalize60s.csv")
subjects = df['Subject'].values
X_all = df.iloc[:, 2:2562].values
X_full = df.iloc[:, 2:-1].values
df = df[df['Label'].isin(['baseline', 'amusement', 'stress'])]
df['EmotionBinary'] = df['Label'].map(lambda x: 0 if x in ['baseline', 'amusement'] else 1)
y_emotion = df['EmotionBinary'].values

# ============ HRV 特徵萃取 ============
hrv_cache = "D:/WESAD_output/hrv_cache.npy"
if os.path.exists(hrv_cache):
    X_hrv_all = np.load(hrv_cache)
else:
    def extract_hrv_features_from_array(X, sampling_rate=256):
        hrv_features, hrv_columns = [], None
        for signal in X:
            try:
                signal = np.array(signal, dtype=np.float32)
                _, rpeaks = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
                hrv_time = nk.hrv_time(rpeaks, sampling_rate=sampling_rate, show=False)
                hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=sampling_rate, show=False)
                hrv_all = pd.concat([hrv_time, hrv_freq], axis=1)
                if hrv_columns is None:
                    hrv_columns = hrv_all.columns
                hrv_features.append(hrv_all.iloc[0].values)
            except:
                nan_array = np.full(len(hrv_columns) if hrv_columns is not None else 50, np.nan)
                hrv_features.append(nan_array)
        hrv_features = np.array(hrv_features)
        remove = ['HRV_ULF','HRV_VLF','HRV_SDANN1','HRV_SDNNI1','HRV_SDANN2','HRV_SDNNI2','HRV_SDANN5','HRV_SDNNI5']
        if hrv_columns is not None:
            indices = [i for i, col in enumerate(hrv_columns) if col in remove]
            hrv_features = np.delete(hrv_features, indices, axis=1)
        return hrv_features
    X_hrv_all = extract_hrv_features_from_array(X_full)
    np.save(hrv_cache, X_hrv_all)

# Dataset
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(np.array(y), dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ============ CNN BaseModel 和 EmotionClassifier ============
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, 32, padding=15), nn.ReLU(), nn.MaxPool1d(8, 2),
            nn.Conv1d(32, 64, 16, padding=7), nn.ReLU(), nn.MaxPool1d(8, 2),
            nn.Conv1d(64, 128, 8, padding=3), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
    def forward(self, x):
        return self.feature_extractor(x).view(x.size(0), -1)

class EmotionClassifier(nn.Module):
    def __init__(self, base, num_classes=2):
        super().__init__()
        self.base = base
        self.classifier = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, num_classes))
    def forward(self, x):
        return self.classifier(self.base(x))

# ============ Fusion Model ============
class FusionClassifier(nn.Module):
    def __init__(self, base_os, base_ts, hrv_dim, num_classes=2):
        super().__init__()
        self.base_os = base_os
        self.base_ts = base_ts
        self.fc = nn.Sequential(
            nn.Linear(128*2 + hrv_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x_ecg, x_hrv):
        feat_os = self.base_os(x_ecg)
        feat_ts = self.base_ts(x_ecg)
        x_comb = torch.cat([feat_os, feat_ts, x_hrv], dim=1)
        return self.fc(x_comb)


# Evaluation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
all_preds, all_labels = [], []
all_preds_os, all_preds_hrv, all_preds_ts = [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
    print(f"\n=== Fold {fold+1} ===")

    X_train_emo, X_test_emo = X_all[train_idx], X_all[test_idx]
    y_train_emo, y_test_emo = y_emotion[train_idx], y_emotion[test_idx]
    X_train_hrv, X_test_hrv = X_hrv_all[train_idx], X_hrv_all[test_idx]

    scaler = StandardScaler()
    X_train_hrv = scaler.fit_transform(X_train_hrv)
    X_test_hrv = scaler.transform(X_test_hrv)

    # 將 HRV 特徵轉換為 Tensor
    X_train_hrv_tensor = torch.tensor(X_train_hrv, dtype=torch.float32)
    X_test_hrv_tensor = torch.tensor(X_test_hrv, dtype=torch.float32)
    X_train_emo_tensor = torch.tensor(X_train_emo, dtype=torch.float32).unsqueeze(1)
    X_test_emo_tensor = torch.tensor(X_test_emo, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train_emo, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_emo, dtype=torch.long)

    # 建立 TensorDataset 與 DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_emo_tensor, X_train_hrv_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_emo_tensor, X_test_hrv_tensor, y_test_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


    # OneStep Model
    base_os = BaseModel().to(device)
    model_os = EmotionClassifier(base_os).to(device)
    model_os.load_state_dict(torch.load(f"saved_models/onestep_10fold_2class/fold{fold+1}_best.pth", weights_only=True))
    base_os = model_os.base
    for param in base_os.parameters():
        param.requires_grad = False
    base_os.eval()

    # TwoStep Model
    base_ts = BaseModel().to(device)
    model_ts = EmotionClassifier(base_ts).to(device)
    model_ts.load_state_dict(torch.load(f"saved_models/twostep_10fold_2class/emotion_fold{fold+1}.pth", weights_only=True))
    base_ts = model_ts.base
    for param in base_ts.parameters():
        param.requires_grad = False
    base_ts.eval()

    model = FusionClassifier(base_os, base_ts, X_train_hrv.shape[1]).to(device)

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 訓練融合分類器
    model.train()
    for epoch in range(100):
        for ecg_batch, hrv_batch, label_batch in train_loader:
            ecg_batch, hrv_batch, label_batch = ecg_batch.to(device), hrv_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            output = model(ecg_batch, hrv_batch)
            loss = criterion(output, label_batch)
            loss.backward()
            optimizer.step()

    # 測試
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for ecg_batch, hrv_batch, label_batch in test_loader:
            ecg_batch, hrv_batch = ecg_batch.to(device), hrv_batch.to(device)
            output = model(ecg_batch, hrv_batch)
            pred = output.argmax(1).cpu().numpy()
            preds.extend(pred)
            labels.extend(label_batch.numpy())

    all_preds.extend(preds)
    all_labels.extend(labels)

def print_per_class_accuracy(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    print("Per-class Accuracy:")
    for i, acc in enumerate(acc_per_class):
        name = f"Class {i}" if class_names is None else class_names[i]
        print(f"  {name}: {acc:.2%}")

# 最終結果
print("\n=== Final LOSO Results (Feature Fusion from Pretrained + Trained FC) ===")
print("Accuracy:", 100 * accuracy_score(all_labels, all_preds))
print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds))
print_per_class_accuracy(all_labels, all_preds, class_names=["non-stress", "stress"])
