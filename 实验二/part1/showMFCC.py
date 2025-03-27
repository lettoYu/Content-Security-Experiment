import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# 加载音频文件
audio_path = "input.wav"
y, sr = librosa.load(audio_path, sr=None)

# 提取MFCC特征
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print("MFCC特征形状:", mfccs.shape)

# 提取声谱图特征
spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
print("声谱图特征形状:", spectrogram.shape)


plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.tight_layout()
plt.show()

# 可视化MFCC特征
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

# 可视化声谱图
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.tight_layout()
plt.show()
