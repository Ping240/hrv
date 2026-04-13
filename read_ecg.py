import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv

class read_data_of_one_subject:
    def __init__(self, path, subject):
        self.keys = ['label', 'subject', 'signal']
        self.signal_keys = ['chest', 'wrist']
        self.chest_sensor_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        self.wrist_sensor_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        os.chdir(path)
        os.chdir(subject)
        with open(subject + '.pkl', 'rb') as file:
            data = pickle.load(file, encoding='latin1') 
        self.data = data

    def get_labels(self):
        return self.data[self.keys[0]]
    
    def get_wrist_data(self):
        assert subject == self.data[self.keys[1]]
        signal = self.data[self.keys[2]]
        wrist_data = signal[self.signal_keys[1]]
        return wrist_data
    
    def get_chest_data(self):
        signal = self.data[self.keys[2]]
        chest_data = signal[self.signal_keys[0]]
        return chest_data


data_set_path = "D:/pincode/hrv/WESAD/"
output_dir = "D:/WESAD_output/"
os.makedirs(output_dir, exist_ok=True)
subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
# subject_ids = [2]
obj_data={}

def save_segments_to_csv(segments, label, subject, output_file, is_first_subject):
    if not segments:
        raise ValueError("Segments list is empty.")
    
    # Assuming all segments have the same length, get the number of features from the first segment
    num_features = len(segments[0])
    
    # Create header with subject, label, and feature names
    header = 'Subject,Label,' + ','.join([f'{i}' for i in range(num_features)])
    
    mode = 'w' if is_first_subject else 'a'
    with open(os.path.join(output_dir, output_file), mode) as f:
        if is_first_subject:
            f.write(header + '\n')  # Write the header
        for segment in segments:
            flat_segment = segment.flatten() if isinstance(segment, np.ndarray) else np.array(segment).flatten()
            row = [subject, label] + flat_segment.tolist()
            f.write(','.join(map(str, row)) + '\n')  # Write each row
            

output_file = 'ecg_all_segments.csv'
is_first_subject = True

for subject_id in subject_ids:
    subject = f'S{subject_id}'
    obj_data[subject] = read_data_of_one_subject(data_set_path, subject)
    chest_data_dict = obj_data[subject].get_chest_data()
    chest_dict_length = {key: len(value) for key, value in chest_data_dict.items()}
    print(f'Subject {subject} chest data lengths: {chest_dict_length}')

    wrist_data_dict = obj_data[subject].get_wrist_data()
    wrist_dict_length = {key: len(value) for key,value in wrist_data_dict.items()}
    print(wrist_dict_length)

    ecg_data = chest_data_dict['ECG']
    # bvp_data = wrist_data_dict['BVP']
    # print(bvp_data)

    original_label_sampling_rate = 700
    target_sampling_rate = 700
    ecg_sampling_rate = 700

    labels = obj_data[subject].get_labels()
    # 計算原始和目標數據點數量
    duration = len(labels) / original_label_sampling_rate  # 數據持續時間（秒）
    target_num_samples = int(duration * target_sampling_rate)  # 目標數據點數量

    # 使用最近鄰方法重新取樣
    original_indices = np.arange(len(labels))
    target_indices = np.linspace(0, len(labels) - 1, target_num_samples)
    downsampled_labels = np.interp(target_indices, original_indices, labels, left=None, right=None, period=None).astype(int)

    print(f"Resampled labels length: {len(downsampled_labels)}")

    # 計算對應的 BVP 索引
    def get_ecg_indices(label_indices, label_sampling_rate, ecg_sampling_rate):
        return np.array([int(idx * ecg_sampling_rate / label_sampling_rate) for idx in label_indices])

    # plt.plot(chest_data_dict['ECG'][0:5000])
    # plt.show()

    # plt.plot(wrist_data_dict['ACC'][0:5000])
    # plt.show()

    

    # unique_labels = set(labels)
    # print(unique_labels)
    baseline = np.asarray([idx for idx, val in enumerate(downsampled_labels) if val == 1])
    stress = np.asarray([idx for idx, val in enumerate(downsampled_labels) if val == 2])
    amusement = np.asarray([idx for idx, val in enumerate(downsampled_labels) if val == 3])
    print(len(baseline))
    print(len(stress))
    print(len(amusement))


    # print("Baseline:", chest_data_dict['ECG'][baseline].shape)

    # 根據 baseline, stress, amusement 的索引抓取對應的 BVP 資料
    baseline_indices = get_ecg_indices(baseline, target_sampling_rate, ecg_sampling_rate)
    stress_indices = get_ecg_indices(stress, target_sampling_rate, ecg_sampling_rate)
    amusement_indices = get_ecg_indices(amusement, target_sampling_rate, ecg_sampling_rate)

    baseline_ecg = ecg_data[baseline_indices]
    stress_ecg = ecg_data[stress_indices]
    amusement_ecg = ecg_data[amusement_indices]

    # plt.plot(baseline_bvp[0:640])
    # plt.show()

    # plt.plot(stress_bvp[0:640])
    # plt.show()

    # plt.plot(amusement_bvp[0:640])
    # plt.show()

    print("Baseline ECG data shape:", baseline_ecg.shape)
    print("Stress ECG data shape:", stress_ecg.shape)
    print("Amusement ECG data shape:", amusement_ecg.shape)

    # 將資料切成 30 秒一段（假設每秒 700 個資料點）
    segment_length = 30 * ecg_sampling_rate

    def split_into_segments(data, segment_length):
        num_segments = len(data) // segment_length
        return [data[i * segment_length:(i + 1) * segment_length] for i in range(num_segments)]

    baseline_segments = split_into_segments(baseline_ecg, segment_length)
    stress_segments = split_into_segments(stress_ecg, segment_length)
    amusement_segments = split_into_segments(amusement_ecg, segment_length)

    # 將每段資料存成 CSV 檔案，一列一列地寫
    # def save_segments_to_csv(segments, label, subject):
    #     num_features = len(segments[0])
    #     header = ','.join([f'{i}' for i in range(num_features)])
    #     with open(os.path.join(output_dir, f'{subject}_{label}_segments.csv'), 'w') as f:
    #         f.write(header + '\n')
    #         for segment in segments:
    #             np.savetxt(f, segment.reshape(1, -1), delimiter=',', fmt='%f')
            
    
    save_segments_to_csv(baseline_segments, 'baseline', subject, output_file, is_first_subject)
    save_segments_to_csv(stress_segments, 'stress', subject, output_file, is_first_subject=False)
    save_segments_to_csv(amusement_segments, 'amusement', subject, output_file, is_first_subject=False)
    
    # 標記第一個 subject 的寫入已完成
    is_first_subject = False

    print(f"Saved {len(baseline_segments)} baseline segments, {len(stress_segments)} stress segments, and {len(amusement_segments)} amusement segments for subject {subject}.")