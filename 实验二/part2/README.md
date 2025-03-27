# UrbanSound8K 实验说明

## 项目简介
本项目展示了如何利用 UrbanSound8K 数据集进行音频分类:
- 训练脚本 (Train.py) 使用 CNN 处理经过对数 Mel 变换的音频特征。
- 测试脚本 (test.py) 加载预训练模型，对输入音频进行推理并输出预测结果。

## 环境要求
- Python 3.7+  
- PyTorch (包含 torchaudio)  
- pandas 用于读取 CSV 文件  
- 其余依赖可根据需要自行安装

## 使用方式
1. 将 UrbanSound8K 数据集存放在指定路径 (训练和测试脚本中的 `audio_dir`、`metadata_csv`)。
2. 运行 Train.py 进行模型训练，默认会在当前目录生成 saved_model.pth。
3. 运行 test.py，指定音频和模型路径，即可进行推理：
   ```
   python test.py --audio <音频文件> --model_path <模型路径>
   ```

## 文件结构
- Train.py：训练集加载、模型定义、训练过程及模型保存。  
- test.py：加载模型并对指定音频文件进行推理。

## 版权信息
- UrbanSound8K 数据集版权归原作者所有，仅供学习与研究使用。
