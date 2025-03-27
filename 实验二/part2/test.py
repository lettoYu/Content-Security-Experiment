import argparse
import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

# 1) 定义与训练时相同的模型结构
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

def main():
    # 2) 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="dog_bark.wav", help="要推理的音频文件路径")
    parser.add_argument("--model_path", type=str, default="saved_model.pth", help="已保存模型的路径")
    args = parser.parse_args()
    
    audio_path = args.audio
    model_path = args.model_path
    
    # 3) 如果需要，你可以在这里定义一个 classID -> class_name 的映射
    #    根据你自己的数据集实际顺序进行调整
    class_id_to_name = {
        0: 'air_conditioner',
        1: 'car_horn',
        2: 'children_playing',
        3: 'dog_bark',
        4: 'drilling',
        5: 'engine_idling',
        6: 'gun_shot',
        7: 'jackhammer',
        8: 'siren',
        9: 'street_music'
    }

    # 4) 定义与训练时相同的音频预处理
    sample_rate = 22050  # 训练时使用的采样率
    n_mels = 64
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
    db_transform = torchaudio.transforms.AmplitudeToDB()
    
    # 5) 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=10).to(device)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 6) 加载音频文件 example.wav
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"未找到音频文件: {audio_path}")
    
    waveform, sr = torchaudio.load(audio_path)
    # 如果多声道，只取第一个通道
    if waveform.shape[0] > 1:
        waveform = waveform[0].unsqueeze(0)
    
    # 如果实际采样率与训练采样率不符，可重采样
    # 例如: if sr != sample_rate: waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    
    # 7) 转为 log-Mel 特征
    with torch.no_grad():
        features = mel_transform(waveform)  # [1, n_mels, time]
        features = db_transform(features)    # 对数尺度
        # 需要扩展一个 batch 维度 => [batch=1, 1, n_mels, time]
        features = features.unsqueeze(0).to(device)
        
        # 8) 模型推理
        output = model(features)  # [1, 10]
        # 取最大值索引作为预测类别
        predicted_id = output.argmax(dim=1).item()
        predicted_class = class_id_to_name.get(predicted_id, f"Unknown_{predicted_id}")
        
        # (可选) 查看预测概率
        probs = F.softmax(output, dim=1)
        predicted_prob = probs[0, predicted_id].item()
    
    # 9) 打印结果
    print(f"音频文件: {audio_path}")
    print(f"预测类别ID: {predicted_id}")
    print(f"预测类别名: {predicted_class}")
    print(f"预测概率: {predicted_prob:.4f}")
    print("预测概率分布:")
    for i, prob in enumerate(probs.squeeze().tolist()):
        class_name = class_id_to_name.get(i, f"Unknown_{i}")
        print(f"{class_name}: {prob:.4f}")
if __name__ == '__main__':
    main()
