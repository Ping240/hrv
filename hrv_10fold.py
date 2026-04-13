import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import neurokit2 as nk
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set_seed(42)
set_seed(99)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ========== 提取 HRV 特徵 ==========
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
            print(f"R-peaks count: {len(rpeaks['ECG_R_Peaks'])}")  #R-peak在正常範圍

            # 合併
            hrv_all = pd.concat([hrv_time, hrv_freq], axis=1)
            
            if hrv_columns is None:
                hrv_columns = hrv_all.columns  # 儲存欄位名

            hrv_features.append(hrv_all.iloc[0].values)
        except Exception as e:
            print(f"Warning at index {i}: {e}")
            print(f"Signal length: {len(signal)}")
            nan_array = np.empty(len(hrv_columns) if hrv_columns is not None else 50)  # 50 是保底估值
            nan_array[:] = np.nan
            hrv_features.append(nan_array)

    hrv_features = np.array(hrv_features)

    # 指定要排除的欄位
    # cols_to_remove = ['HRV_ULF', 'HRV_VLF', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2', 
    #                   'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5']
    cols_to_remove = ['HRV_ULF', 'HRV_VLF', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2', 
                      'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5','HRV_LF', 'HRV_LFHF', 'HRV_LFn']
    if hrv_columns is not None:
        remove_indices = [i for i, col in enumerate(hrv_columns) if col in cols_to_remove]
        hrv_features = np.delete(hrv_features, remove_indices, axis=1)
        hrv_columns = [col for col in hrv_columns if col not in cols_to_remove]

    return hrv_features, hrv_columns


# ========== 讀取數據 ==========
df = pd.read_csv("D:/pincode/hrv/WESAD_output/subjectwise_zscore_normalize60s.csv")
subjects = df['Subject'].unique()


# 提取 HRV 特徵（以每列作為一段 ECG 信號）
X_ecg_raw = df.iloc[:, 2:-1].values
df = df[df['Label'].isin(['baseline', 'amusement', 'stress'])]
df['EmotionBinary'] = df['Label'].map(lambda x: 0 if x in ['baseline', 'amusement'] else 1)
y_emotion = df['EmotionBinary'].values

X_hrv, hrv_columns = extract_hrv_features_from_array(X_ecg_raw)
###
print(f"提取的 HRV 特徵形狀: {X_hrv.shape}")
print(X_hrv)

# 计算整个数组中的 NaN 总数
nan_total = np.isnan(X_hrv).sum()
print(f"NaN 值总数: {nan_total}")

# 计算每列中的 NaN 数量（可选）
nan_per_column = np.isnan(X_hrv).sum(axis=0)
print("\n每列 NaN 数量:")
for col_idx, count in enumerate(nan_per_column):
    print(f"列 {col_idx}: {count} 个 NaN")

print(f"欄特徵名稱: {hrv_columns}")
###
def main():
    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
    kf = KFold(n_splits=10, shuffle=True, random_state=99)
    os.makedirs("saved_models/hrv_10fold_2class", exist_ok=True)

    all_preds, all_labels = [], []
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_hrv, y_emotion), 1):
        print(f"\n=== Fold {fold}/10 ===")

        X_train, X_test = X_hrv[train_idx], X_hrv[test_idx]
        y_train, y_test = y_emotion[train_idx], y_emotion[test_idx]

        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            # random_state=42
            random_state=99
        )

        clf.fit(X_train_normalized, y_train)

        train_preds = clf.predict(X_train_normalized).ravel()
        test_preds = clf.predict(X_test_normalized).ravel()

        # 訓練集評估
        train_acc = 100 * np.mean(train_preds == y_train)
        print(f"Train Acc: {train_acc:.2f}%")

        # 測試集評估
        test_acc = 100 * np.mean(test_preds == y_test)
        print(f"Fold {fold} Accuracy: {test_acc:.2f}%")

        # 一定要累積，overall 才算得出來
        fold_accuracies.append(test_acc)
        all_preds.extend(test_preds.tolist())
        all_labels.extend(y_test.tolist())

        # 儲存模型
        import pickle
        model_path = f"saved_models/hrv_10fold_2class/fold{fold}_best.pkl"
        scaler_path = f"saved_models/hrv_10fold_2class/fold{fold}_scaler.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        print(f"儲存模型至: {model_path}")

    # 總體結果
    print("\n=== Overall 10-Fold 結果 ===")
    print(f"平均準確率: {np.mean(fold_accuracies):.2f}%")
    print("混淆矩陣:\n", confusion_matrix(all_labels, all_preds))
    print("分類報告:\n", classification_report(
        all_labels,
        all_preds,
        target_names=['Baseline/Amusement', 'Stress'],
        zero_division=0
    ))
if __name__ == "__main__":
    # Generate synthetic signals
    ecg = nk.ecg_simulate(duration=10, heart_rate=70)
    data = pd.DataFrame({"ECG": ecg})
    nk.signal_plot(data, subplots=True)
    main()