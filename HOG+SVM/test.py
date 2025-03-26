import cv2
from PIL import Image
import numpy as np

# 加载训练好的SVM模型
svm = cv2.ml.SVM_load("hog_svm_pedestrian.xml")

# 初始化HOG特征提取器
hog = cv2.HOGDescriptor()

def fix_png(img_path):
    try:
        img = Image.open(img_path)
        img.save(img_path, "PNG")
    except Exception as e:
        print(f"PNG修复失败: {str(e)}")

# HOG特征计算函数
def compute_hog_features(img_path):
    fix_png(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")
    img = cv2.resize(img, (64, 128))
    return hog.compute(img).flatten().reshape(1, -1).astype(np.float32)

# 测试图片路径
test_img_path = "example.png"

# 计算特征
feature = compute_hog_features(test_img_path)

# 模型预测
_, prediction = svm.predict(feature)

# 输出结果
if prediction[0][0] == 1:
    print("检测结果：图片中【存在行人】")
else:
    print("检测结果：图片中【不存在行人】")
