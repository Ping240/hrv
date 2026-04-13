# [file name]: normalize_and_save.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def per_segment_normalize(X):
    """按每个样本（段）独立标准化"""
    X_norm = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    return X_norm

def subjectwise_zscore(df, signal_cols, subject_col='Subject'):
    df_norm = df.copy()
    for subject in df[subject_col].unique():
        idx = df[subject_col] == subject
        subj_data = df.loc[idx, signal_cols]
        mean = subj_data.mean()
        std = subj_data.std()
        df_norm.loc[idx, signal_cols] = (subj_data - mean) / std
    return df_norm

# def standardize(X):
#     """标准化函数"""
#     X_mean = np.mean(X)
#     X_std = np.std(X)
#     return (X - X_mean) / X_std


def process_and_save():
    # 读取原始数据
    # df = pd.read_csv("D:/pincode/hrv/WESAD_output/data_with_gender60s.csv")
    df = pd.read_csv("D:/pincode/hrv/WESAD_output/data_with_gender60s_700hz.csv")
    # 提取特征列和标签列
    signal_cols = df.columns[2:-1].tolist()  # 假设特征从第3列到倒数第二列
    meta_cols = df.columns[:2].tolist()      # 假设前两列是元数据（如Subject）
    label_col = df.columns[-1]               # 最后一列是标签

    # # 提取特征数据并进行标准化
    # X_all = df[signal_cols].values
    # X_normalized = standardize(X_all)
    df_normalized = subjectwise_zscore(df, signal_cols, subject_col='Subject')
    X_normalized = df_normalized[signal_cols].values

    # 重建DataFrame
    df_normalized = pd.DataFrame(
        data=X_normalized,
        columns=signal_cols
    )
    df_normalized[meta_cols] = df[meta_cols]  # 添加元数据列
    df_normalized[label_col] = df[label_col]  # 添加标签列

    # 按原始列顺序排序
    ordered_columns = meta_cols + signal_cols + [label_col]
    df_normalized = df_normalized[ordered_columns]

    # 保存标准化后的数据
    df_normalized.to_csv("D:/pincode/hrv/WESAD_output/subjectwise_zscore_normalize60s_700hz.csv", index=False)
    print("标准化数据已保存为 subjectwise_zscore_normalize_700hz.csv")
    # df_normalized.to_csv("D:/pincode/hrv/WESAD_output/subjectwise_zscore_normalize60s.csv", index=False)
    # print("标准化数据已保存为 subjectwise_zscore_normalize.csv")

if __name__ == "__main__":
    process_and_save()