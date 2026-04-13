import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 數據讀取 ============
df = pd.read_csv("D:/WESAD_output/subjectwise_zscore_normalize60s.csv")
label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(df['Label'].values)
X_all = df.iloc[:, 2:2562].values

# ============ Dataset 類 ============
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(np.array(y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============ 模型架構 ============
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

class EmotionClassifier(nn.Module):
    def __init__(self, base_model, num_classes=3):
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

# ============ 10-Fold 交叉驗證 ============
if __name__ == "__main__":
    os.makedirs("saved_models/onestep_10fold", exist_ok=True)

    all_preds, all_labels = [], []
    fold_accuracies = []

    skf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), 1):
        print(f"\n=== Fold {fold}/10 ===")

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        train_dataset = ECGDataset(X_train, y_train)
        test_dataset = ECGDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        base_model = BaseModel().to(device)
        model = EmotionClassifier(base_model).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        best_state_dict = None

        model.train()
        for epoch in range(200):
            total_loss, correct = 0, 0
            for ecg, labels in train_loader:
                ecg, labels = ecg.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(ecg)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * ecg.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
            acc = 100 * correct / len(train_dataset)
            if acc > best_acc:
                best_acc = acc
                best_state_dict = model.state_dict()
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}: Loss={total_loss/len(train_dataset):.4f}, Acc={acc:.2f}%")

        # 儲存最佳模型
        torch.save(best_state_dict, f"saved_models/onestep_10fold/fold{fold}_best.pth")
        print(f"儲存模型至: saved_models/onestep_10fold/fold{fold}_best.pth (Best Train Acc: {best_acc:.2f}%)")

        # 測試階段
        model.load_state_dict(best_state_dict)
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for ecg, y in test_loader:
                ecg = ecg.to(device)
                outputs = model(ecg)
                pred = outputs.argmax(1).cpu().numpy()
                label = y.numpy()
                preds.extend(pred)
                labels.extend(label)

        all_preds.extend(preds)
        all_labels.extend(labels)
        acc = 100 * np.mean(np.array(preds) == np.array(labels))
        fold_accuracies.append(acc)
        print(f"Fold {fold} Accuracy: {acc:.2f}%")

    # 總體結果
    print("\n=== Overall 10-Fold 結果 ===")
    print(f"平均準確率: {np.mean(fold_accuracies):.2f}%")
    print("混淆矩陣:\n", confusion_matrix(all_labels, all_preds))
    print("分類報告:\n", classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
