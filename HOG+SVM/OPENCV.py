from PIL import Image
import cv2
import numpy as np
import os
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# 忽略 OpenCV PNG 相关警告
warnings.filterwarnings("ignore", category=UserWarning, module="cv2")
warnings.filterwarnings("ignore", message=".*libpng warning: iCCP: known incorrect sRGB profile.*")

# HOG 特征提取器
hog = cv2.HOGDescriptor()

def fix_png(img_path):
    """ 用 PIL 重新保存 PNG，去除错误的 sRGB profile """
    try:
        img = Image.open(img_path)
        img.save(img_path, "PNG")  # 重新保存去除错误的 sRGB
    except Exception as e:
        print(f"处理 PNG 失败: {img_path} - {str(e)}")

def compute_hog_features(img_path):
    """ 计算 HOG 特征 """
    fix_png(img_path)  # 修复 PNG 图片
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return None
    img = cv2.resize(img, (64, 128))
    return hog.compute(img).flatten()

def load_dataset(positive_path, negative_path):
    """ 加载数据集 """
    pos_features = [compute_hog_features(os.path.join(positive_path, f)) for f in os.listdir(positive_path) if f.endswith('.png')]
    neg_features = [compute_hog_features(os.path.join(negative_path, f)) for f in os.listdir(negative_path) if f.endswith('.png')]

    # 过滤掉 None（避免损坏图像）
    pos_features = [f for f in pos_features if f is not None]
    neg_features = [f for f in neg_features if f is not None]

    if not pos_features or not neg_features:
        raise ValueError("数据集为空，请检查数据路径！")

    X = np.array(pos_features + neg_features, dtype=np.float32)
    y = np.array([1] * len(pos_features) + [0] * len(neg_features), dtype=np.int32)
    return X, y

# 训练 SVM
positive_path = "INRIAPerson/Train/pos"
negative_path = "INRIAPerson/Train/neg"

X, y = load_dataset(positive_path, negative_path)

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 SVM
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(0.01)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3))
svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

# 评估模型
_, y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"模型评估结果：")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# 保存 SVM 模型
svm.save("hog_svm_pedestrian.xml")

print("SVM 训练完成，模型已保存！")
