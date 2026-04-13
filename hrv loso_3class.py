import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import neurokit2 as nk
import torch
import torch.nn as nn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


            # 合併
            hrv_all = pd.concat([hrv_time, hrv_freq], axis=1)
            
            if hrv_columns is None:
                hrv_columns = hrv_all.columns  # 儲存欄位名

            hrv_features.append(hrv_all.iloc[0].values)
        except Exception as e:
            print(f"Warning at index {i}: {e}")
            nan_array = np.empty(len(hrv_columns) if hrv_columns is not None else 50)  # 50 是保底估值
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


# ========== 讀取數據 ==========
df = pd.read_csv("D:/WESAD_output/subjectwise_zscore_normalize60s.csv")
subjects = df['Subject'].unique()
loso = LeaveOneGroupOut()

# 提取 HRV 特徵（以每列作為一段 ECG 信號）
X_ecg_raw = df.iloc[:, 2:-1].values
y_emotion = LabelEncoder().fit_transform(df['Label'].values)
groups = df['Subject'].values

X_hrv, hrv_columns = extract_hrv_features_from_array(X_ecg_raw)


# ========== LOSO 主流程 ==========
def main():
    emotion_accuracies = []
    all_preds, all_labels = [], []

    for fold, (train_idx, test_idx) in enumerate(loso.split(X_hrv, y_emotion, groups)):
        print(f"\n=== Fold {fold + 1}: Subject {subjects[fold]} ===")

        # Split
        X_train, X_test = X_hrv[train_idx], X_hrv[test_idx]
        y_train, y_test = y_emotion[train_idx], y_emotion[test_idx]

        # 2. Z-score標準化 - 使用訓練集計算的均值和標準差
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)  # 使用訓練集計算的均值和標準差標準化測試集

        # clf = SVC(kernel='linear', class_weight='balanced', random_state=42)

        # clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

        # 線性判別分析（LDA）
        # clf = LinearDiscriminantAnalysis()

        # # 隨機森林
        # clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

        # 決策樹
        # clf = DecisionTreeClassifier(class_weight='balanced', random_state=42)

        # # AdaBoost（使用決策樹作為基礎學習器）
        # clf = AdaBoostClassifier(n_estimators=50,random_state=42)

        # # K-最近鄰（KNN）
        clf = KNeighborsClassifier(n_neighbors=5, weights='distance') 

        clf.fit(X_train_normalized, y_train)

        # Predict
        preds = clf.predict(X_test_normalized)
        acc = 100 * np.mean(preds == y_test)
        emotion_accuracies.append(acc)
        all_preds.extend(preds)
        all_labels.extend(y_test)

        # Evaluation
        print(f"Emotion Test Accuracy: {acc:.2f}%")
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
        print("Classification Report:\n", classification_report(
            y_test, preds, zero_division=0
        ))

    def print_per_class_accuracy(y_true, y_pred, class_names=None):
        cm = confusion_matrix(y_true, y_pred)
        acc_per_class = cm.diagonal() / cm.sum(axis=1)
        print("Per-class Accuracy:")
        for i, acc in enumerate(acc_per_class):
            name = f"Class {i}" if class_names is None else class_names[i]
            print(f"  {name}: {acc:.2%}")

    # Summary
    print("\n=== LOSO Summary ===")
    print(f"Average Emotion Accuracy: {np.mean(emotion_accuracies):.2f}% ± {np.std(emotion_accuracies):.2f}%")

    # 方法二：總體 accuracy
    overall_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall LOSO Accuracy (combined): {overall_accuracy * 100:.2f}%")

    print("\nOverall Emotion Results:")
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("Classification Report:\n", classification_report(
        all_labels, all_preds, zero_division=0
    ))
    print_per_class_accuracy(all_labels, all_preds, class_names=["amusement", "baseline", "stress"])


    # Plot accuracy curve
    # plt.figure(figsize=(6, 4))
    # plt.plot(emotion_accuracies, label='Emotion Acc', marker='o', color='green')
    # plt.title('Emotion Accuracy per Fold')
    # plt.xlabel('Fold')
    # plt.ylabel('Accuracy (%)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('hrv_randomforest_loso_accuracy.png')
    # plt.show()

if __name__ == "__main__":
    main()
