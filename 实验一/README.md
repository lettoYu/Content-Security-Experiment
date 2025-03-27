# 内容安全实验一

本实验包含两个主要部分：基于 YOLOv10 的目标检测和基于 HOG+SVM 的行人检测。

---

## 实验内容

### YOLOv10
- **目标**：使用 [YOLOv10](https://github.com/THU-MIG/yolov10) 进行目标检测。
- **工具**：数据集下载工具 [Coco-to-yolo-downloader](https://github.com/maldivien/Coco-to-yolo-downloader)。
- **功能**：提供数据集格式转换、模型训练与测试脚本。

### HOG+SVM
- **目标**：使用 HOG 特征与 SVM 分类器进行行人检测。
- **数据集**：[INRIAPerson](https://drive.google.com/file/d/1wTxod2BhY_HUkEdDYRVSuw-nDuqrgCu7/view)。

---

## 运行说明

### YOLOv10
1. **数据集准备**：
   - 使用 `main.py` 下载数据集。
   - 运行 `dataprocess.py` 转换数据集格式。
2. **模型训练**：
   - 运行 `my_train.py` 进行模型训练。
3. **模型测试**：
   - 运行 `test.py` 进行模型测试。

### HOG+SVM
1. **检测**：
   - 运行 `OPENCV.py` 进行行人检测。
2. **测试**：
   - 运行 `test.py` 测试示例图片 `example.png`。

---

## 更多细节

请参阅各子目录中的 `README.md` 文件以获取更详细的运行说明和实验结果。