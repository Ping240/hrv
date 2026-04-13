# ✅ 改為 LOSO 並載入 one-step 權重，不重新訓練

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
import torch.nn.functional as F
import neurokit2 as nk
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 設定種子與裝置
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HRV 特徵提取
def extract_hrv_features_from_array(X, sampling_rate=256):
    hrv_features = []
    hrv_columns = None
    for i, signal in enumerate(X):
        try:
            signal = np.array(signal, dtype=np.float32)
            _, rpeaks = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
            hrv_time = nk.hrv_time(rpeaks, sampling_rate=sampling_rate, show=False)
            hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=sampling_rate, show=False)
            hrv_all = pd.concat([hrv_time, hrv_freq], axis=1)
            if hrv_columns is None:
                hrv_columns = hrv_all.columns
            hrv_features.append(hrv_all.iloc[0].values)
        except Exception as e:
            nan_array = np.empty(len(hrv_columns) if hrv_columns is not None else 50)
            nan_array[:] = np.nan
            hrv_features.append(nan_array)
    hrv_features = np.array(hrv_features)
    cols_to_remove = ['HRV_ULF', 'HRV_VLF', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2', 
                      'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5']
    if hrv_columns is not None:
        remove_indices = [i for i, col in enumerate(hrv_columns) if col in cols_to_remove]
        hrv_features = np.delete(hrv_features, remove_indices, axis=1)
    return hrv_features

# 載入資料
df = pd.read_csv("D:/WESAD_output/subjectwise_zscore_normalize60s.csv")
subjects = df['Subject'].values
X = df.iloc[:, 2:-1].values
X_emotion = df.iloc[:, 2:2562].values
X_hrv_all = extract_hrv_features_from_array(X)
y_emotion = LabelEncoder().fit_transform(df['Label'].values)

# Dataset class
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(np.array(y).astype(np.int64), dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# 模型定義
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, 32, padding=15), nn.ReLU(), nn.MaxPool1d(8, 2),
            nn.Conv1d(32, 64, 16, padding=7), nn.ReLU(), nn.MaxPool1d(8, 2),
            nn.Conv1d(64, 128, 8, padding=3), nn.ReLU(), nn.AdaptiveMaxPool1d(1)
        )
    def forward(self, x): return self.feature_extractor(x).view(x.size(0), -1)

class EmotionClassifier(nn.Module):
    def __init__(self, base_model, num_classes=3):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes))
    def forward(self, x): return self.classifier(self.base(x))

# LOSO 測試流程（不重新訓練 one-step，直接載入權重）
def loso_test_with_saved_onestep():
    logo = LeaveOneGroupOut()
    all_preds, all_labels = [], []
    all_preds_hrv, all_preds_os, all_preds_fusion = [], [], []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X_emotion, y_emotion, groups=subjects)):
        subject = subjects[test_idx[0]]
        print(f"\n===== Fold {fold+1} - Subject {subject} =====")
        X_train_emo, X_test_emo = X_emotion[train_idx], X_emotion[test_idx]
        y_train_emo, y_test_emo = y_emotion[train_idx], y_emotion[test_idx]
        X_train_hrv, X_test_hrv = X_hrv_all[train_idx], X_hrv_all[test_idx]

        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train_hrv)
        X_test_normalized = scaler.transform(X_test_hrv)

        # 載入 one-step 權重模型
        base = BaseModel().to(device)
        model = EmotionClassifier(base).to(device)
        weight_path = f"saved_models/onestep/emotion_{subject}.pth"
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        model.eval()

        def get_probs(model, data):
            loader = DataLoader(ECGDataset(data, np.zeros(len(data))), batch_size=128)
            all_probs = []
            with torch.no_grad():
                for ecg, _ in loader:
                    out = model(ecg.to(device))
                    all_probs.append(F.softmax(out, dim=1).cpu().numpy())
            return np.vstack(all_probs)

        probs_os = get_probs(model, X_test_emo)
        preds_os = probs_os.argmax(1)

        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_train_normalized, y_train_emo)
        probs_rf = rf.predict_proba(X_test_normalized)
        preds_rf = probs_rf.argmax(1)

        # fusion
        probs_os_train = get_probs(model, X_train_emo)
        probs_rf_train = rf.predict_proba(X_train_normalized)
        # meta_clf = LogisticRegression(max_iter=1000)
        meta_clf = LogisticRegression(max_iter=1000, random_state=42)
       
        meta_clf.fit(np.hstack([probs_os_train, probs_rf_train]), y_train_emo)
        probs_fuse_test = np.hstack([probs_os, probs_rf])
        preds_fuse = meta_clf.predict(probs_fuse_test)

        print("Accuracy (Fusion):", 100 * accuracy_score(y_test_emo, preds_fuse))
        print("Accuracy (One-Step):", 100 * accuracy_score(y_test_emo, preds_os))
        print("Accuracy (HRV RF):", 100 * accuracy_score(y_test_emo, preds_rf))

        all_preds.extend(preds_fuse)
        all_labels.extend(y_test_emo)
        all_preds_os.extend(preds_os)
        all_preds_hrv.extend(preds_rf)

    print("\n=== Final LOSO Results (Fusion) ===")
    print("Overall Accuracy:", 100 * accuracy_score(all_labels, all_preds))
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds))

    print("\n=== Ablation: One-Step Only ===")
    print("Overall Accuracy (One-Step):", 100 * accuracy_score(all_labels, all_preds_os))
    print(confusion_matrix(all_labels, all_preds_os))
    print(classification_report(all_labels, all_preds_os))

    print("\n=== Ablation: HRV RF Only ===")
    print("Overall Accuracy (HRV RF):", 100 * accuracy_score(all_labels, all_preds_hrv))
    print(confusion_matrix(all_labels, all_preds_hrv))
    print(classification_report(all_labels, all_preds_hrv))

if __name__ == '__main__':
    loso_test_with_saved_onestep()
