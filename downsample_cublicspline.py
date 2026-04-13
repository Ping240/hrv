import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import CubicSpline

# 讀取 CSV 檔案
file_path = "D:/pincode/hrv/WESAD_output/ecg_all_segments.csv"
data = pd.read_csv(file_path)

# 選擇資料框中的特定列
X_resampled = data.iloc[:, 2:]
labels = data['Label']
subjects = data['Subject']

# 原始採樣率和目標採樣率
original_sampling_rate = 700
target_sampling_rate = 256

# 計算下採樣後的樣本數
num_samples = int(X_resampled.shape[1] * (target_sampling_rate / original_sampling_rate))

# 初始化一個空的 DataFrame 來存儲下採樣後的數據
downsampled_data = pd.DataFrame()

# 下採樣每一行數據，並轉換回 Pandas Series
for index in range(len(X_resampled)):
    row = X_resampled.iloc[index, :].values  # 將 pandas.Series 轉換為 numpy 陣列
    time_original = np.linspace(0, len(row) / original_sampling_rate, len(row), endpoint=False)
    time_downsampled = np.linspace(0, len(row) / original_sampling_rate, num_samples, endpoint=False)
    
    # 使用三次樣條插值進行下採樣
    cs = CubicSpline(time_original, row)
    downsampled_signal = cs(time_downsampled)
    
    # 將 'Subject'、'Label' 和訊號組成新一行
    new_row = [subjects.iloc[index], labels.iloc[index]] + downsampled_signal.tolist()
    downsampled_data = pd.concat([downsampled_data, pd.DataFrame([new_row])], ignore_index=True)

# 設定新的欄位名稱
new_columns = ['Subject', 'Label'] + [f'{i}' for i in range(num_samples)]
downsampled_data.columns = new_columns

# 保存下採樣後的數據到新的 CSV 文件
downsampled_data.to_csv("D:/pincode/hrv/WESAD_output/ecg_downsampled_256hz_60s.csv", index=False)

# 繪製下採樣後的第一條資料
plt.figure(figsize=(12, 6))

# 原始信號
plt.subplot(2, 1, 1)
plt.plot(time_original, X_resampled.iloc[0, :].values, label='700Hz Signal')
plt.title('First Row of X_resampled (Original Signal)')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.grid()

# 下採樣後的信號
plt.subplot(2, 1, 2)
plt.plot(time_downsampled, downsampled_data.iloc[0, 2:].values, label='256Hz Downsampled Signal', color='red')
plt.title('First Row of X_downsampled (Downsampled Signal)')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.grid()

plt.tight_layout()
plt.show()