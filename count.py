import pandas as pd

# 讀取資料（請根據實際檔案路徑調整）
df = pd.read_csv("D:/pincode/hrv/WESAD_output/subjectwise_zscore_normalize60s.csv")

# 計算每位受測者在每種情緒下的筆數
count_per_subject_label = df.groupby(['Subject', 'Label']).size().unstack(fill_value=0)

# 印出結果
print(count_per_subject_label)
