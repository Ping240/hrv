import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 固定CUDA卷积算法
    torch.backends.cudnn.benchmark = False     # 关闭自动优化

# 在训练前调用
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================== 1️⃣ 数据准备 ========================
# 读取ECG数据
df = pd.read_csv("D:/pincode/hrv/WESAD_output/ecg_all_segments.csv")

# X_all = df.iloc[:, 2:-1].values

# 第一阶段数据（性别分类）
X_gender = df.iloc[:, 2:2562].values
y_gender = df['Gender'].values

# 第二阶段数据（情感分类）
X_emotion = df.iloc[:, 2:2562].values
df = df[df['Label'].isin(['baseline', 'amusement', 'stress'])]
df['EmotionBinary'] = df['Label'].map(lambda x: 0 if x in ['baseline', 'amusement'] else 1)
y_emotion = df['EmotionBinary'].values

# ======================== 2️⃣ 数据预处理 ========================
# 性别数据划分
X_train_gen, X_test_gen, y_train_gen, y_test_gen = train_test_split(
    X_gender, y_gender, test_size=0.1, random_state=42
)

# 情感数据划分
X_train_emo, X_test_emo, y_train_emo, y_test_emo = train_test_split(
    X_emotion, y_emotion, test_size=0.1, random_state=42
)

# ======================== 2️⃣ 修改数据集类 ========================
class ECGDataset(Dataset):
    def __init__(self, X, y, task='gender'):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        
        # 确保标签数据为数值类型
        if isinstance(y, (np.ndarray, list)):
            y = np.array(y).astype(np.int64 if task=='emotion' else np.float32)
            
        self.y = torch.tensor(
            y, 
            dtype=torch.long if task=='emotion' else torch.float32
        )
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ======================== 4️⃣ 模型架构 ========================
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, 32, padding=15),
            nn.ReLU(),
            # nn.Conv1d(32, 32, 32, padding=15),
            # nn.ReLU(),
            nn.MaxPool1d(8, 2),
            nn.Conv1d(32, 64, 16, padding=7),
            nn.ReLU(),
            # nn.Conv1d(64, 64, 16, padding=7),
            # nn.ReLU(),
            nn.MaxPool1d(8, 2),
            nn.Conv1d(64, 128, 8, padding=3),
            nn.ReLU(),
            # nn.Conv1d(128, 128, 8, padding=3),
            # nn.ReLU(),
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
    def __init__(self, base_model, num_classes=2):
        super().__init__()
        self.base = base_model
        for param in self.base.parameters():  # 冻结特征提取器
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

# ======================== 5️⃣ 第一阶段训练（性别分类）====================
def train_gender_classifier():
    # 初始化
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_gender  = BaseModel().to(device)
    gender_model = GenderClassifier(base_model_gender).to(device)
    optimizer = optim.Adam(gender_model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 数据加载
    train_dataset = ECGDataset(X_train_gen, y_train_gen)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 训练循环
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
            # 计算准确率
            preds = (torch.sigmoid(outputs)) > 0.5
            correct += (preds == labels).sum().item()
            total_loss += loss.item()

        train_loss = total_loss/len(train_loader)
        train_acc = 100 * correct / len(train_dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
    
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(gender_model.state_dict(), "gender_model.pth")
            torch.save(base_model_gender.state_dict(), "gender_base.pth")

    # 绘制图表（仅训练集）
    plt.figure(figsize=(12, 5))
    
    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc', color='red')
    plt.title('Training Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('train_curves.png')
    plt.show()

# ======================== 6️⃣ 第二阶段训练（情感分类）====================
def train_emotion_classifier():
    # 初始化
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载预训练模型
    base_model_emotion = BaseModel().to(device)
    base_model_emotion.load_state_dict(torch.load("gender_base.pth", weights_only=True))
    
    emotion_model = EmotionClassifier(base_model_emotion).to(device)
    optimizer = optim.Adam(emotion_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 数据加载
    train_dataset = ECGDataset(X_train_emo, y_train_emo, task='emotion')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 训练循环
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
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.2f}%")
    
        if acc > best_acc:
            best_acc = acc
            torch.save(emotion_model.state_dict(), "emotion_clf.pth")

    # 绘制图表（仅训练集）
    plt.figure(figsize=(12, 5))
    
    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc', color='red')
    plt.title('Training Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('train_curves.png')
    plt.show()

# ======================== 新增测试函数 ========================
def test_gender_classifier():
    # 初始化
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载预训练模型
    base_model_gender = BaseModel().to(device)
    gender_model = GenderClassifier(base_model_gender).to(device)
    gender_model.load_state_dict(torch.load("gender_model.pth", weights_only=True))
    
    # 创建测试数据集
    test_dataset = ECGDataset(X_test_gen, y_test_gen)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 评估指标
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
    
    # 生成评估报告
    print("\n=== Gender Test Results ===")
    print("Accuracy:", 100 * np.mean(np.array(all_preds) == np.array(all_labels)))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("Classification Report:\n", classification_report(
        all_labels, 
        all_preds, 
        target_names=['Female', 'Male']  # 根据实际标签顺序调整
    ))

# ======================== 7️⃣ 测试评估 ========================
def test_emotion_classifier():
    # 初始化
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练基础模型
    base_model_emotion = BaseModel().to(device)
    base_model_emotion.load_state_dict(torch.load("gender_base.pth", weights_only=True))
    
    # 加载情感分类器
    emotion_model = EmotionClassifier(base_model_emotion).to(device)
    emotion_model.load_state_dict(torch.load("emotion_clf.pth", weights_only=True))
    emotion_model.eval()

    # 创建测试数据集
    test_dataset = ECGDataset(X_test_emo, y_test_emo, task='emotion')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 评估指标
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for ecg, labels in test_loader:
            ecg = ecg.to(device)
            labels = labels.cpu().numpy()
            
            outputs = emotion_model(ecg)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)

    # 生成评估报告
    print("\n=== Emotion Test Results ===")
    print("Accuracy:", 100 * np.mean(np.array(all_preds) == np.array(all_labels)))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("Classification Report:\n", classification_report(
        all_labels, 
        all_preds, 
    ))

# ======================== 执行训练 ========================
if __name__ == "__main__":
    # 第一阶段训练
    print("=== Training Gender Classifier ===")
    train_gender_classifier()

    # 第一阶段测试
    print("\n=== Testing Gender Classifier ===")
    test_gender_classifier()
    
    # 第二阶段训练
    print("\n=== Training Emotion Classifier ===")
    train_emotion_classifier()

    # 测试情感分类器
    print("\n=== Testing Emotion Classifier ===")
    test_emotion_classifier()