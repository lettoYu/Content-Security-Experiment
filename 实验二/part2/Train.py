import os
import random
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ------------------------------
# UrbanSound8K 数据集定义
# ------------------------------
class UrbanSoundDataset(Dataset):
    def __init__(self, metadata_csv, audio_dir, max_samples_per_class=20, transform=None):
        """
        Args:
            metadata_csv (str): CSV 文件路径（包含 fold、slice_file_name、classID 等信息）
            audio_dir (str): 音频数据所在目录（例如 "UrbanSound8K/audio"）
            max_samples_per_class (int): 每个类别选取的最大样本数
            transform (callable, optional): 对 waveform 进行处理的变换（例如 MelSpectrogram+AmplitudeToDB）
        """
        self.metadata = pd.read_csv(metadata_csv)
        # 为每个类别随机选取少量样本
        selected_indices = []
        for cls in sorted(self.metadata['classID'].unique()):
            cls_idx = self.metadata[self.metadata['classID'] == cls].index.tolist()
            sampled = random.sample(cls_idx, min(max_samples_per_class, len(cls_idx)))
            selected_indices.extend(sampled)
        self.metadata = self.metadata.loc[selected_indices].reset_index(drop=True)
        self.audio_dir = audio_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 获取当前数据的元信息
        row = self.metadata.iloc[idx]
        fold = row['fold']
        file_name = row['slice_file_name']
        class_id = row['classID']
        # 构造音频文件完整路径：UrbanSound8K/audio/fold{fold}/{file_name}
        file_path = os.path.join(self.audio_dir, f"fold{fold}", file_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")
        
        # 加载音频
        waveform, sample_rate = torchaudio.load(file_path)
        # 如果多通道，只取第一个通道
        if waveform.shape[0] > 1:
            waveform = waveform[0].unsqueeze(0)
        
        # 应用预处理变换（例如提取 log-Mel spectrogram）
        if self.transform:
            features = self.transform(waveform)
        else:
            features = waveform
        # 返回特征与标签
        return features, class_id

# ------------------------------
# 自定义 collate_fn：对 batch 内的样本在时间维度进行零填充
# ------------------------------
def pad_collate(batch):
    """
    对每个样本的特征（形状：[1, n_mels, time]）在 time 维度上做零填充，使得所有样本的长度相同
    """
    features, labels = zip(*batch)
    max_len = max(feat.shape[-1] for feat in features)
    padded_features = []
    for feat in features:
        pad_size = max_len - feat.shape[-1]
        padded = F.pad(feat, (0, pad_size))  # 在 time 维度右侧填充
        padded_features.append(padded)
    features = torch.stack(padded_features)
    labels = torch.tensor(labels)
    return features, labels

# ------------------------------
# 简单的 CNN 模型
# ------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ------------------------------
# 训练函数
# ------------------------------
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (batch_idx + 1) % 5 == 0:
            print(f"Epoch [{epoch}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    print(f"Epoch [{epoch}] Average Loss: {running_loss / len(train_loader):.4f}")

# ------------------------------
# 评估函数
# ------------------------------
def evaluate(model, device, data_loader, criterion, mode="Validation"):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item()
            pred = outputs.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    avg_loss = test_loss / len(data_loader)
    accuracy = 100. * correct / total
    print(f"{mode} - Avg Loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    return avg_loss, accuracy

# ------------------------------
# 主函数：训练、评估和保存模型
# ------------------------------
def main():
    # 参数设置
    metadata_csv = "UrbanSound8K/metadata/UrbanSound8K.csv"
    audio_dir = "UrbanSound8K/audio"
    max_samples_per_class = 20  # 每个类别选取20个样本
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001

    # 预处理：MelSpectrogram + AmplitudeToDB
    sample_rate = 22050  # 请根据实际数据调整
    n_mels = 64
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
    db_transform = torchaudio.transforms.AmplitudeToDB()
    transform = nn.Sequential(mel_transform, db_transform)

    # 创建数据集
    dataset = UrbanSoundDataset(metadata_csv, audio_dir, max_samples_per_class, transform=transform)
    # 随机划分训练集和验证集（80%/20%）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建 DataLoader，使用自定义 collate_fn 保证张量尺寸一致
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    
    # 设备设置（使用 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    
    # 初始化模型、损失函数和优化器
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练和评估
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch} ---")
        train(model, device, train_loader, optimizer, criterion, epoch)
        evaluate(model, device, val_loader, criterion, mode="Validation")
    
    # 训练完成后保存模型参数
    torch.save(model.state_dict(), "saved_model.pth")
    print("\n训练和评估完成，模型已保存到 saved_model.pth")

if __name__ == '__main__':
    main()
