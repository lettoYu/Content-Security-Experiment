import os
import shutil
import random

# 源路径
pos_dir = 'C2Y/annotations/pos'
neg_dir = 'C2Y/annotations/neg'

# 目标路径
out_dir = 'datasets'
image_train_dir = os.path.join(out_dir, 'images/train')
image_val_dir = os.path.join(out_dir, 'images/val')
label_train_dir = os.path.join(out_dir, 'labels/train')
label_val_dir = os.path.join(out_dir, 'labels/val')

# 创建输出文件夹
for d in [image_train_dir, image_val_dir, label_train_dir, label_val_dir]:
    os.makedirs(d, exist_ok=True)

# 收集所有图片-标签对
all_pairs = []
for folder in [pos_dir, neg_dir]:
    for file in os.listdir(folder):
        if file.endswith(('.jpg', '.png')):
            base = os.path.splitext(file)[0]
            img_path = os.path.join(folder, file)
            label_path = os.path.join(folder, f'{base}.txt')
            if os.path.exists(label_path):
                all_pairs.append((img_path, label_path))

# 随机打乱并划分
random.shuffle(all_pairs)
split = int(len(all_pairs) * 0.8)
train_pairs = all_pairs[:split]
val_pairs = all_pairs[split:]

def copy_files(pairs, img_dst, label_dst):
    for img_path, label_path in pairs:
        name = os.path.basename(img_path)
        base = os.path.splitext(name)[0]
        shutil.copy(img_path, os.path.join(img_dst, name))
        shutil.copy(label_path, os.path.join(label_dst, f'{base}.txt'))

copy_files(train_pairs, image_train_dir, label_train_dir)
copy_files(val_pairs, image_val_dir, label_val_dir)

print(f"完成：训练 {len(train_pairs)} 张，验证 {len(val_pairs)} 张。")
