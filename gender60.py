import pandas as pd
import numpy as np

# 1. 读取数据
# df = pd.read_csv("D:/pincode/hrv//WESAD_output/ecg_downsampled_256hz_60s.csv")
df = pd.read_csv("D:/pincode/hrv//WESAD_output/ecg_all_segments.csv")
# 2. 定义女性受试者列表
female_subjects = ["S8", "S11", "S17"]

# 3. 处理subject列格式（可选）
df["Subject"] = df["Subject"].astype(str).str.strip().str.upper()

# 4. 新增gender列
df["Gender"] = np.where(df["Subject"].isin(female_subjects), "0", "1")

print(df)  # 打印前5行以检查结果

# 5. 保存结果
df.to_csv("D:/pincode/hrv/WESAD_output/data_with_gender60s_700hz.csv", index=False)