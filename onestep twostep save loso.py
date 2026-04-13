# LOSO Version of Two-Step Training with Per-Subject Best Weight Saving + Combined Evaluation
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
import torch.nn.functional as F
import neurokit2 as nk
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
subjects = df['Subject'].values
X_all = df.iloc[:, 2:2562].values
X_full = df.iloc[:, 2:-1].values
label_encoder = LabelEncoder()
y_emotion = label_encoder.fit_transform(df['Label'].values)
y_gender = df['Gender'].values


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
    def __init__(self, base, num_classes=3):
        super().__init__()
        self.base = base
        self.classifier = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, num_classes))
    def forward(self, x): return self.classifier(self.base(x))

# Gating Network
# class GatingNet(nn.Module):
#     def __init__(self, input_dim=9):
#         super().__init__()
#         self.gate = nn.Sequential(
#             nn.Linear(input_dim, 64), nn.ReLU(),
#             nn.Linear(64, 3), nn.Softmax(dim=1))  # output weights for 3 models
#     def forward(self, x): return self.gate(x)

# Run LOSO with OneStep + HRV + TwoStep fusion
logo = LeaveOneGroupOut()
all_preds_fuse, all_labels = [], []
all_preds_os, all_preds_hrv, all_preds_ts = [], [], []

for fold, (train_idx, test_idx) in enumerate(logo.split(X_all, y_emotion, groups=subjects)):
    subject = subjects[test_idx[0]]
    print(f"\n=== Fold {fold+1} - Subject {subject} ===")

    X_train_emo, X_test_emo = X_all[train_idx], X_all[test_idx]
    y_train_emo, y_test_emo = y_emotion[train_idx], y_emotion[test_idx]

    # HRV normalization
    # scaler = StandardScaler()
    # X_train_hrv = scaler.fit_transform(X_train_hrv)
    # X_test_hrv = scaler.transform(X_test_hrv)

    # Load OneStep model
    base = BaseModel().to(device)
    model_os = EmotionClassifier(base).to(device)
    model_os.load_state_dict(torch.load(f"saved_models/onestep/emotion_{subject}.pth", weights_only=True))
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

    # Load TwoStep model
    base_ts = BaseModel().to(device)
    model_ts = EmotionClassifier(base_ts).to(device)
    model_ts.load_state_dict(torch.load(f"saved_models/twostep/emotion_{subject}.pth", weights_only=True))
    model_ts.eval()

    probs_ts = get_probs(model_ts, X_test_emo)
    preds_ts = probs_ts.argmax(1)

    # HRV model
    # clf_hrv = LinearDiscriminantAnalysis()
    # clf_hrv = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    # clf_hrv.fit(X_train_hrv, y_train_emo)
    # probs_hrv = clf_hrv.predict_proba(X_test_hrv)
    # preds_hrv = probs_hrv.argmax(1)

    # Meta Fusion
    probs_os_train = get_probs(model_os, X_train_emo)
    probs_ts_train = get_probs(model_ts, X_train_emo)
    # probs_hrv_train = clf_hrv.predict_proba(X_train_hrv)

    # meta_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    # meta_clf = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, loss_function='MultiClass', verbose=0, random_state=42)
    # meta_clf = CatBoostClassifier(random_state=42)
    
    # meta_clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
    # meta_clf = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
    # meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    # meta_clf.fit(np.hstack([probs_os_train, probs_ts_train]), y_train_emo)
    # probs_fuse_test = np.hstack([probs_os, probs_ts])
    # preds_fuse = meta_clf.predict(probs_fuse_test)

    # Equal Weighted Fusion (1/3 each)
    # probs_fuse = 0.1 * probs_os + 0.1 * probs_ts + 0.8 * probs_hrv
    # probs_fuse = 0.1 * probs_os + 0.2 * probs_ts + 0.7 * probs_hrv
    # probs_fuse = 1/3 * probs_os + 1/3 * probs_ts + 1/3 * probs_hrv
    # probs_fuse = 0.2 * probs_os + 0.2 * probs_ts + 0.6 * probs_hrv
    # probs_fuse = 0.15 * probs_os + 0.25 * probs_ts + 0.60 * probs_hrv
    probs_fuse = 1/2 * probs_os  + 1/2 * probs_ts
    preds_fuse = probs_fuse.argmax(1)

    # best_acc = 0
    # best_weights = (0.0, 0.0, 1.0)
    # best_preds = None
    # for w_os in np.arange(0, 1.1, 0.1):
    #     for w_ts in np.arange(0, 1.1 - w_os, 0.1):
    #         w_hrv = 1.0 - w_os - w_ts
    #         probs_fuse_temp = w_os * probs_os + w_ts * probs_ts + w_hrv * probs_hrv
    #         preds_temp = probs_fuse_temp.argmax(1)
    #         acc = accuracy_score(y_test_emo, preds_temp)
    #         if acc > best_acc:
    #             best_acc = acc
    #             best_weights = (w_os, w_ts, w_hrv)
    #             best_preds = preds_temp

    # print(f"Best Weights for Subject {subject}: OS={best_weights[0]:.2f}, TS={best_weights[1]:.2f}, HRV={best_weights[2]:.2f} -> Acc={best_acc*100:.2f}%")
    # preds_fuse = best_preds


    all_preds_fuse.extend(preds_fuse)
    all_labels.extend(y_test_emo)
    all_preds_os.extend(preds_os)
    # all_preds_hrv.extend(preds_hrv)
    all_preds_ts.extend(preds_ts)

# Final Report
print("\n=== Final LOSO Results (Fusion) ===")
print("Accuracy:", 100 * accuracy_score(all_labels, all_preds_fuse))
print(confusion_matrix(all_labels, all_preds_fuse))
print(classification_report(all_labels, all_preds_fuse))

print("\n=== Ablation: One-Step ===")
print("Accuracy:", 100 * accuracy_score(all_labels, all_preds_os))
print(confusion_matrix(all_labels, all_preds_os))
print(classification_report(all_labels, all_preds_os))

# print("\n=== Ablation: HRV ===")
# print("Accuracy:", 100 * accuracy_score(all_labels, all_preds_hrv))
# print(confusion_matrix(all_labels, all_preds_hrv))
# print(classification_report(all_labels, all_preds_hrv))

print("\n=== Ablation: Two-Step ===")
print("Accuracy:", 100 * accuracy_score(all_labels, all_preds_ts))
print(confusion_matrix(all_labels, all_preds_ts))
print(classification_report(all_labels, all_preds_ts))
