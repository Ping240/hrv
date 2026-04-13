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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

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
            # print(f"R-peaks count: {len(rpeaks['ECG_R_Peaks'])}")  #R-peak在正常範圍

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
# ========== 90/10 數據分割 ==========
def main():
    # 90/10 分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_hrv, y_emotion, 
        test_size=0.1, 
        random_state=42
    )
    
    print(f"訓練集大小: {X_train.shape[0]} 樣本")
    print(f"測試集大小: {X_test.shape[0]} 樣本")

    # 2. Z-score標準化 - 使用訓練集計算的均值和標準差
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)  # 使用訓練集計算的均值和標準差標準化測試集

    
    # 訓練 Random Forest
    # clf = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, loss_function='MultiClass', verbose=0, random_state=42)
    # clf =SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train_normalized, y_train)

    # 預測
    train_preds = clf.predict(X_train_normalized).ravel()
    test_preds = clf.predict(X_test_normalized).ravel()
    
    # 訓練集評估
    train_acc = 100 * np.mean(train_preds == y_train)
    print(f"\n訓練集準確率: {train_acc:.2f}%")
    print("訓練集混淆矩陣:\n", confusion_matrix(y_train, train_preds))
    
    # 測試集評估
    test_acc = 100 * np.mean(test_preds == y_test)
    print(f"\n測試集準確率: {test_acc:.2f}%")
    print("測試集混淆矩陣:\n", confusion_matrix(y_test, test_preds))
    print("測試集分類報告:\n", classification_report(y_test, test_preds, zero_division=0))


if __name__ == "__main__":
    main()