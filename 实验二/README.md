# 总体项目说明

本目录下包含两个子文件夹：
- part1：展示了如何加载音频文件并提取特征（如波形图、MFCC、声谱图）。
- part2：基于 UrbanSound8K 数据集进行音频分类的示例。

## 快速开始
1. 在 part1 下，修改 `showMFCC.py` 的音频文件路径后运行脚本即可查看可视化结果。
2. 在 part2 下，按需配置依赖并执行训练脚本或测试脚本。
3. 各自文件夹下均有对应的README文件，请参考使用。

## 依赖
- Python 3.7+
- librosa
- matplotlib
- numpy
- PyTorch (part2 用于音频分类)
