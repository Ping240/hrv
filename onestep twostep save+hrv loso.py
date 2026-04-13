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
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.signal import butter, filtfilt
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



# ======================== 濾波器函數 ========================
# def butter_bandpass_filter(signal, lowcut=8.0, highcut=20.0, fs=256.0, order=2):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='bandpass')
#     return filtfilt(b, a, signal)

# Load Data
df = pd.read_csv("D:/WESAD_output/subjectwise_zscore_normalize60s.csv")
subjects = df['Subject'].values
X_all = df.iloc[:, 2:2562].values
X_full = df.iloc[:, 2:-1].values
# X_filtered = np.array([butter_bandpass_filter(x) for x in X_full])

df = df[df['Label'].isin(['baseline', 'amusement', 'stress'])]
df['EmotionBinary'] = df['Label'].map(lambda x: 0 if x in ['baseline', 'amusement'] else 1)
y_emotion = df['EmotionBinary'].values
y_gender = df['Gender'].values

# Extract HRV
hrv_cache = "D:/WESAD_output/hrv_cache.npy"
if os.path.exists(hrv_cache):
    X_hrv_all = np.load(hrv_cache)
else:
    def extract_hrv_features_from_array(X, sampling_rate=256):
        hrv_features = []
        hrv_columns = None

        for i, signal in enumerate(X):
            try:
                signal = np.array(signal, dtype=np.float32)
                _, rpeaks = nk.ecg_peaks(signal, sampling_rate=sampling_rate)

                # 時域特徵
                hrv_time = nk.hrv_time(rpeaks, sampling_rate=sampling_rate, show=False)

                # 頻域特徵
                hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=sampling_rate, show=False)

                # 非線性特徵
                # hrv_nonlinear = nk.hrv_nonlinear(rpeaks, sampling_rate=sampling_rate, show=False)
                # wanted_features = ["HRV_SD1", "HRV_SD2", "HRV_SD1SD2", "HRV_S", "HRV_CSI", "HRV_CVI"]
                # hrv_nonlinear = hrv_nonlinear[[f for f in wanted_features if f in hrv_nonlinear.columns]]

                # 合併
                hrv_all = pd.concat([hrv_time, hrv_freq], axis=1)
                
                if hrv_columns is None:
                    hrv_columns = hrv_all.columns  # 儲存欄位名

                hrv_features.append(hrv_all.iloc[0].values)
            except Exception as e:
                print(f"Warning at index {i}: {e}")
                nan_array = np.empty(len(hrv_columns) if hrv_columns is not None else 100)  # 50 是保底估值
                nan_array[:] = np.nan
                hrv_features.append(nan_array)

        hrv_features = np.array(hrv_features)

        # 指定要排除的欄位
        cols_to_remove = ['HRV_ULF', 'HRV_VLF', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2', 
                        'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5']
        if hrv_columns is not None:
            remove_indices = [i for i, col in enumerate(hrv_columns) if col in cols_to_remove]
            hrv_features = np.delete(hrv_features, remove_indices, axis=1)
            hrv_columns = [col for col in hrv_columns if col not in cols_to_remove]

        return hrv_features, hrv_columns
    X_hrv_all, hrv_columns = extract_hrv_features_from_array(X_full)
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

# # Gating Network
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
    X_train_hrv, X_test_hrv = X_hrv_all[train_idx], X_hrv_all[test_idx]

    # HRV normalization
    scaler = StandardScaler()
    X_train_hrv = scaler.fit_transform(X_train_hrv)
    X_test_hrv = scaler.transform(X_test_hrv)

    # selector = SelectKBest(score_func=f_classif, k=30)
    # X_train_hrv = selector.fit_transform(X_train_hrv, y_train_emo)
    # X_test_hrv = selector.transform(X_test_hrv)

    # Load OneStep model
    base = BaseModel().to(device)
    model_os = EmotionClassifier(base).to(device)
    model_os.load_state_dict(torch.load(f"saved_models/onestep_2class/emotion_{subject}.pth", weights_only=True))
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
    model_ts.load_state_dict(torch.load(f"saved_models/twostep_2class/emotion_{subject}.pth", weights_only=True))
    model_ts.eval()

    probs_ts = get_probs(model_ts, X_test_emo)
    preds_ts = probs_ts.argmax(1)

    # HRV model
    # clf_hrv = LinearDiscriminantAnalysis()
    clf_hrv = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    # clf_hrv = CatBoostClassifier(random_state=42)
    clf_hrv.fit(X_train_hrv, y_train_emo)
    probs_hrv = clf_hrv.predict_proba(X_test_hrv)
    preds_hrv = probs_hrv.argmax(1)

    # Meta Fusion
    probs_os_train = get_probs(model_os, X_train_emo)
    probs_ts_train = get_probs(model_ts, X_train_emo)
    probs_hrv_train = clf_hrv.predict_proba(X_train_hrv)

    # meta_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    # meta_clf = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, loss_function='MultiClass', verbose=0, random_state=42)
    meta_clf = CatBoostClassifier(random_state=42, verbose=0)
    # # meta_clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
    # # meta_clf = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
    # meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    # meta_clf.fit(np.hstack([probs_os_train, probs_ts_train, probs_hrv_train]), y_train_emo)
    # probs_fuse_test = np.hstack([probs_os, probs_ts, probs_hrv])
    
    # preds_fuse = meta_clf.predict(probs_fuse_test)

    # Equal Weighted Fusion (1/3 each)
    # probs_fuse = 0.1 * probs_os + 0.1 * probs_ts + 0.8 * probs_hrv
    # probs_fuse = 0.1 * probs_os + 0.2 * probs_ts + 0.7 * probs_hrv
    probs_fuse = 1/3 * probs_os + 1/3 * probs_ts + 1/3 * probs_hrv
    # probs_fuse = 0.2 * probs_os + 0.2 * probs_ts + 0.6 * probs_hrv
    # probs_fuse = 0.15 * probs_os + 0.25 * probs_ts + 0.60 * probs_hrv
    # probs_fuse = 1/2 * probs_os  + 1/2 * probs_hrv
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
    all_preds_hrv.extend(preds_hrv)
    all_preds_ts.extend(preds_ts)

def print_per_class_accuracy(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    acc_per_class = cm.diagonal() / cm.sum(axis=1)
    print("Per-class Accuracy:")
    for i, acc in enumerate(acc_per_class):
        name = f"Class {i}" if class_names is None else class_names[i]
        print(f"  {name}: {acc:.2%}")

# Final Report
print("\n=== Final LOSO Results (Fusion) ===")
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

