# 10-Fold Version of OneStep + TwoStep + HRV Fusion Evaluation
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
import torch.nn.functional as F
import neurokit2 as nk
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

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

# Load Data
df = pd.read_csv("D:/WESAD_output/subjectwise_zscore_normalize60s.csv")
X_all = df.iloc[:, 2:2562].values
X_full = df.iloc[:, 2:-1].values
df = df[df['Label'].isin(['baseline', 'amusement', 'stress'])]
df['EmotionBinary'] = df['Label'].map(lambda x: 0 if x in ['baseline', 'amusement'] else 1)
y_emotion = df['EmotionBinary'].values

# Extract HRV
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

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, 32, padding=15), nn.ReLU(), nn.MaxPool1d(8, 2),
            nn.Conv1d(32, 64, 16, padding=7), nn.ReLU(), nn.MaxPool1d(8, 2),
            nn.Conv1d(64, 128, 8, padding=3), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
    def forward(self, x): return self.feature_extractor(x).view(x.size(0), -1)

class EmotionClassifier(nn.Module):
    def __init__(self, base, num_classes=2):
        super().__init__()
        self.base = base
        self.classifier = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, num_classes))
    def forward(self, x): return self.classifier(self.base(x))

# Evaluation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
all_preds_fuse, all_labels = [], []
all_preds_os, all_preds_hrv, all_preds_ts = [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
    print(f"\n=== Fold {fold+1} ===")

    X_train_emo, X_test_emo = X_all[train_idx], X_all[test_idx]
    y_train_emo, y_test_emo = y_emotion[train_idx], y_emotion[test_idx]
    X_train_hrv, X_test_hrv = X_hrv_all[train_idx], X_hrv_all[test_idx]

    scaler = StandardScaler()
    X_train_hrv = scaler.fit_transform(X_train_hrv)
    X_test_hrv = scaler.transform(X_test_hrv)

    # OneStep Model
    base_os = BaseModel().to(device)
    model_os = EmotionClassifier(base_os).to(device)
    model_os.load_state_dict(torch.load(f"saved_models/onestep_10fold_2class/fold{fold+1}_best.pth", weights_only=True))
    model_os.eval()

    def get_probs(model, data):
        loader = DataLoader(ECGDataset(data, np.zeros(len(data))), batch_size=128)
        all_probs = []
        with torch.no_grad():
            for ecg, _ in loader:
                out = model(ecg.to(device))
                all_probs.append(F.softmax(out, dim=1).cpu().numpy())
        return np.vstack(all_probs)

    probs_os = get_probs(model_os, X_test_emo)
    preds_os = probs_os.argmax(1)

    # TwoStep Model
    base_ts = BaseModel().to(device)
    model_ts = EmotionClassifier(base_ts).to(device)
    model_ts.load_state_dict(torch.load(f"saved_models/twostep_10fold_2class/emotion_fold{fold+1}.pth", weights_only=True))
    model_ts.eval()

    probs_ts = get_probs(model_ts, X_test_emo)
    preds_ts = probs_ts.argmax(1)

    # HRV Model
    
    clf_hrv = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf_hrv.fit(X_train_hrv, y_train_emo)
    probs_hrv = clf_hrv.predict_proba(X_test_hrv)
    preds_hrv = probs_hrv.argmax(1)

    probs_os_train = get_probs(model_os, X_train_emo)
    probs_ts_train = get_probs(model_ts, X_train_emo)
    probs_hrv_train = clf_hrv.predict_proba(X_train_hrv)

    # meta_clf = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, loss_function='MultiClass', verbose=0, random_state=42)
    # meta_clf = CatBoostClassifier(iterations=300, learning_rate=0.01, depth=10, loss_function='MultiClass', verbose=0, random_state=42)
    # meta_clf = CatBoostClassifier(iterations=300, learning_rate=0.01, depth=6, loss_function='MultiClass', verbose=0, random_state=42)
    
    # meta_clf = CatBoostClassifier(random_state=42)
    # meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    # meta_clf.fit(np.hstack([probs_os_train, probs_ts_train, probs_hrv_train]), y_train_emo)
    # probs_fuse_test = np.hstack([probs_os, probs_ts, probs_hrv])
    
    # preds_fuse = meta_clf.predict(probs_fuse_test)

    # Fusion
    probs_fuse = 1/3 * probs_os + 1/3 * probs_ts + 1/3 * probs_hrv
    # probs_fuse = 1/5 * probs_os  + 1/5 * probs_hrv
    preds_fuse = probs_fuse.argmax(1)

    # Save Results
    all_preds_os.extend(preds_os)
    all_preds_ts.extend(preds_ts)
    all_preds_hrv.extend(preds_hrv)
    all_preds_fuse.extend(preds_fuse)
    all_labels.extend(y_test_emo)

def print_per_class_accuracy(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    print("Per-class Accuracy:")
    for i, acc in enumerate(acc_per_class):
        name = f"Class {i}" if class_names is None else class_names[i]
        print(f"  {name}: {acc:.2%}")

# Final Report
print("\n=== Final 10-Fold Results (Fusion) ===")
print("Accuracy:", 100 * accuracy_score(all_labels, all_preds_fuse))
print(confusion_matrix(all_labels, all_preds_fuse))
print(classification_report(all_labels, all_preds_fuse))
print_per_class_accuracy(all_labels, all_preds_fuse, class_names=["non-stress", "stress"])


print("\n=== Ablation: One-Step ===")
print("Accuracy:", 100 * accuracy_score(all_labels, all_preds_os))
print(confusion_matrix(all_labels, all_preds_os))
print(classification_report(all_labels, all_preds_os))
print_per_class_accuracy(all_labels, all_preds_os, class_names=["non-stress", "stress"])


print("\n=== Ablation: HRV ===")
print("Accuracy:", 100 * accuracy_score(all_labels, all_preds_hrv))
print(confusion_matrix(all_labels, all_preds_hrv))
print(classification_report(all_labels, all_preds_hrv))
print_per_class_accuracy(all_labels, all_preds_hrv, class_names=["non-stress", "stress"])


print("\n=== Ablation: Two-Step ===")
print("Accuracy:", 100 * accuracy_score(all_labels, all_preds_ts))
print(confusion_matrix(all_labels, all_preds_ts))
print(classification_report(all_labels, all_preds_ts))
print_per_class_accuracy(all_labels, all_preds_ts, class_names=["non-stress", "stress"])

