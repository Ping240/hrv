import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================== 1️⃣ 數據準備 ========================
df = pd.read_csv("D:/WESAD_output/subjectwise_zscore_normalize60s.csv")

X_emotion = df.iloc[:, 2:2562].values
df = df[df['Label'].isin(['baseline', 'amusement', 'stress'])]
df['EmotionBinary'] = df['Label'].map(lambda x: 0 if x in ['baseline', 'amusement'] else 1)
y_emotion = df['EmotionBinary'].values

X_gender = df.iloc[:, 2:2562].values
y_gender = df['Gender'].values

# 數據分割
X_train_gen, X_test_gen, y_train_gen, y_test_gen = train_test_split(
    X_gender, y_gender, test_size=0.1, random_state=42
)
X_train_emo, X_test_emo, y_train_emo, y_test_emo = train_test_split(
    X_emotion, y_emotion, test_size=0.1, random_state=42
)

#===================================================================
import neurokit2 as nk
import pandas as pd
import numpy as np

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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
X_signals = df.iloc[:, 2:-1].values
X_hrv, hrv_titles = extract_hrv_features_from_array(X_signals, sampling_rate=256)
# print(X_hrv.shape)
# print(hrv_titles)

# 假設 HRV 特徵為 X_hrv，情緒標籤為 y_emotion
X_hrv_train, X_hrv_test, y_hrv_train, y_hrv_test = train_test_split(
    X_hrv, y_emotion, test_size=0.1, random_state=42
)

# HRV normalization
scaler = StandardScaler()
X_train_hrv = scaler.fit_transform(X_hrv_train)
X_test_hrv = scaler.transform(X_hrv_test)

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_hrv, y_hrv_train)

# 預測類別與機率
y_pred_dt = rf.predict(X_test_hrv)
probs_dt = rf.predict_proba(X_test_hrv)


# 評估
acc_rf = accuracy_score(y_hrv_test, y_pred_dt)
print("Random Forest HRV Test Accuracy: {:.2f}%".format(acc_rf * 100))
print("Confusion Matrix:\n", confusion_matrix(y_hrv_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_hrv_test, y_pred_dt))

# ======================== 2️⃣ 數據集類 ========================
class ECGDataset(Dataset):
    def __init__(self, X, y, task='gender'):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        if isinstance(y, (np.ndarray, list)):
            y = np.array(y).astype(np.int64 if task == 'emotion' else np.float32)
        self.y = torch.tensor(
            y, 
            dtype=torch.long if task == 'emotion' else torch.float32
        )
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ======================== 3️⃣ 模型架構 ========================
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, 32, padding=15),
            nn.ReLU(),
            nn.MaxPool1d(8, 2),
            nn.Conv1d(32, 64, 16, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(8, 2),
            nn.Conv1d(64, 128, 8, padding=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)

class GenderClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        features = self.base(x)
        return self.classifier(features)

class EmotionClassifier(nn.Module):
    def __init__(self, base_model, num_classes=2, freeze_base=True):
        super().__init__()
        self.base = base_model
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.base(x)
        return self.classifier(features)

class EmotionClassifierOneStep(nn.Module):
    def __init__(self, base_model, num_classes=2):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.base(x)
        return self.classifier(features)

# ======================== 4️⃣ 訓練性別分類器 ========================
def train_gender_classifier():
    base_model_gender = BaseModel().to(device)
    gender_model = GenderClassifier(base_model_gender).to(device)
    optimizer = optim.Adam(gender_model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = ECGDataset(X_train_gen, y_train_gen)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    train_losses = []
    train_accuracies = []
    best_acc = 0
    num_epochs = 100

    for epoch in range(num_epochs):
        gender_model.train()
        total_loss, correct = 0, 0
        for ecg, labels in train_loader:
            ecg = ecg.to(device)
            labels = labels.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = gender_model(ecg)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = (torch.sigmoid(outputs) > 0.5)
            correct += (preds == labels).sum().item()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / len(train_dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        print(f"Gender Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
    
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(gender_model.state_dict(), "gender_model.pth")
            torch.save(base_model_gender.state_dict(), "gender_base.pth")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.title('Gender Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc', color='red')
    plt.title('Gender Training Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('gender_train_curves.png')
    plt.show()

# ======================== 5️⃣ 訓練情感分類器（第一個模型） ========================
def train_emotion_classifier():
    base_model_emotion = BaseModel().to(device)
    base_model_emotion.load_state_dict(torch.load("gender_base.pth", weights_only=True))
    
    emotion_model = EmotionClassifier(base_model_emotion).to(device)
    optimizer = optim.Adam(emotion_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_dataset = ECGDataset(X_train_emo, y_train_emo, task='emotion')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    num_epochs = 600
    best_acc = 0
    train_losses = []
    train_accuracies = []
    for epoch in range(num_epochs):
        emotion_model.train()
        total_loss = 0
        correct = 0
        for ecg, labels in train_loader:
            ecg = ecg.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = emotion_model(ecg)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        
        acc = 100 * correct / len(train_dataset)
        loss = total_loss / len(train_loader)
        train_losses.append(loss)
        train_accuracies.append(acc)
        print(f"Emotion (Two-step) Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.2f}%")
    
        if acc > best_acc:
            best_acc = acc
            torch.save(emotion_model.state_dict(), "emotion_clf_two_step.pth")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.title('Emotion (Two-step) Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc', color='red')
    plt.title('Emotion (Two-step) Training Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('emotion_two_step_train_curves.png')
    plt.show()

# ======================== 6️⃣ 訓練情感分類器（第二個模型，獨立訓練） ========================
def train_emotion_classifier_one_step():
    base_model_emotion = BaseModel().to(device)
    emotion_model = EmotionClassifierOneStep(base_model_emotion).to(device)
    optimizer = optim.Adam(emotion_model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_dataset = ECGDataset(X_train_emo, y_train_emo, task='emotion')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    num_epochs = 200
    best_acc = 0
    train_losses = []
    train_accuracies = []
    for epoch in range(num_epochs):
        emotion_model.train()
        total_loss = 0
        correct = 0
        for ecg, labels in train_loader:
            ecg = ecg.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = emotion_model(ecg)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        
        acc = 100 * correct / len(train_dataset)
        loss = total_loss / len(train_loader)
        train_losses.append(loss)
        train_accuracies.append(acc)
        print(f"Emotion (One-step) Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.2f}%")
    
        if acc > best_acc:
            best_acc = acc
            torch.save(emotion_model.state_dict(), "emotion_clf_one_step.pth")
            torch.save(base_model_emotion.state_dict(), "base_emotion_one_step.pth")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.title('Emotion (One-step) Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc', color='red')
    plt.title('Emotion (One-step) Training Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('emotion_one_step_train_curves.png')
    plt.show()

# ======================== 7️⃣ 測試性別分類器 ========================
def test_gender_classifier():
    base_model_gender = BaseModel().to(device)
    gender_model = GenderClassifier(base_model_gender).to(device)
    gender_model.load_state_dict(torch.load("gender_model.pth", weights_only=True))
    
    test_dataset = ECGDataset(X_test_gen, y_test_gen)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    gender_model.eval()
    with torch.no_grad():
        for ecg, labels in test_loader:
            ecg = ecg.to(device)
            outputs = gender_model(ecg)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().astype(int))
    
    print("\n=== Gender Test Results ===")
    print("Accuracy:", 100 * np.mean(np.array(all_preds) == np.array(all_labels)))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("Classification Report:\n", classification_report(
        all_labels, 
        all_preds, 
        target_names=['Female', 'Male']
    ))

# ======================== 8️⃣ 測試情感分類器（融合） ========================
from sklearn.linear_model import LogisticRegression

def test_emotion_classifier_fusion():
    # 1. 载入模型（保持不变）
    base_model_two_step = BaseModel().to(device)
    base_model_two_step.load_state_dict(torch.load("gender_base.pth", weights_only=True))
    emotion_model_two_step = EmotionClassifier(base_model_two_step).to(device)
    emotion_model_two_step.load_state_dict(torch.load("emotion_clf_two_step.pth", weights_only=True))
    
    base_model_one_step = BaseModel().to(device)
    base_model_one_step.load_state_dict(torch.load("base_emotion_one_step.pth", weights_only=True))
    emotion_model_one_step = EmotionClassifierOneStep(base_model_one_step).to(device)
    emotion_model_one_step.load_state_dict(torch.load("emotion_clf_one_step.pth", weights_only=True))

    # 2. 在训练集上获取概率特征
    def get_probs(model, data):
        dataset = ECGDataset(data, np.zeros(len(data)), task='emotion')
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        probs = []
        model.eval()
        with torch.no_grad():
            for ecg, _ in loader:
                ecg = ecg.to(device)
                outputs = model(ecg)
                probs.append(F.softmax(outputs, dim=1).cpu().numpy())
        return np.vstack(probs)
    
    # 获取训练集概率
    train_probs_two_step = get_probs(emotion_model_two_step, X_train_emo)
    train_probs_one_step = get_probs(emotion_model_one_step, X_train_emo)
    train_probs_hrv = rf.predict_proba(X_hrv_train)  # HRV模型训练集概率
    
    # 拼接概率特征 (N_samples, 9)
    X_fusion_train = np.hstack([
        train_probs_two_step,
        train_probs_one_step,
        train_probs_hrv
    ])
    
    # 3. 训练元分类器
    meta_classifier = LogisticRegression(
        # multi_class='multinomial',
        max_iter=1000,
        random_state=42
    )
    
    meta_classifier.fit(X_fusion_train, y_train_emo)
    
    # # 4. 在测试集上融合预测
    # test_dataset = ECGDataset(X_test_emo, y_test_emo, task='emotion')
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    test_probs_two_step = get_probs(emotion_model_two_step, X_test_emo)
    test_probs_one_step = get_probs(emotion_model_one_step, X_test_emo)
    test_probs_hrv = probs_dt  # 已有的HRV测试集概率
    
    # 拼接测试集概率特征
    X_fusion_test = np.hstack([
        test_probs_two_step,
        test_probs_one_step,
        test_probs_hrv
    ])
    
    # # 元分类器预测
    y_pred_fusion = meta_classifier.predict(X_fusion_test)

    # y_prob_fusion = 1/3 * test_probs_one_step + 1/3 * test_probs_two_step + 1/3 * test_probs_hrv
    # y_pred_fusion =  y_prob_fusion.argmax(1)

    # 5. 评估结果
    print("\n=== Emotion Test Results (Probability Fusion) ===")
    print("Accuracy:", 100 * np.mean(y_pred_fusion == y_test_emo))
    print("Confusion Matrix:\n", confusion_matrix(y_test_emo, y_pred_fusion))
    print("Classification Report:\n", classification_report(
        y_test_emo, 
        y_pred_fusion, 
        target_names=['Baseline/Amusement', 'Stress']
    ))

# ======================== 執行訓練與測試 ========================
if __name__ == "__main__":
    # print("=== Training Gender Classifier ===")
    # train_gender_classifier()

    # print("\n=== Testing Gender Classifier ===")
    # test_gender_classifier()
    
    # print("\n=== Training Emotion Classifier (Two-step) ===")
    # train_emotion_classifier()

    # print("\n=== Training Emotion Classifier (One-step) ===")
    # train_emotion_classifier_one_step()

    print("\n=== Testing Emotion Classifier (Fusion) ===")
    test_emotion_classifier_fusion()