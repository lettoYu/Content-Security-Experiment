# YOLO步骤

## 1. 数据集准备

### 1.1 数据集格式
YOLO使用VOC数据集格式，数据集目录结构如下：
```
VOCdevkit/
    VOC2007/
        Annotations/
        ImageSets/
            Main/
                train.txt
                val.txt
                trainval.txt
                test.txt
        JPEGImages/
    VOC2012/
        Annotations/
        ImageSets/
            Main/
                train.txt
                val.txt
                trainval.txt
                test.txt
        JPEGImages/
```
其中，Annotations目录存放的是xml格式的标签文件，ImageSets目录存放的是txt格式的标签文件，JPEGImages目录存放的是图片文件。

### 1.2 数据集下载

使用 C2Y 中的main.py下载数据集，其中下载的内容指定为categories_to_download.txt中的类别,包含名称及ID(指定的)，下载的数据集会保存在./annotations/目录下。

### 1.3. 数据集处理

返回上级目录，运行dataprocess.py，将数据集转换为YOLO格式，并生成训练集和验证集的txt文件。
编写对应的YAML文件。

## 2. 模型训练

使用my_train.py进行模型训练，训练过程中会自动保存模型。

## 3. 模型测试

使用test.py进行模型测试，测试过程中会自动保存测试结果。example.jpg为测试图片，test.txt为测试结果。

