# 遍历当前文件夹下的图片
# 验证结果
import hashlib
from PIL import Image
from collections import defaultdict
import os
import shutil
import random
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm


def tqdm_file_count(dest_dir):
    # 获取文件数量
    filecounter = 0
    for _, _, files in tqdm(os.walk(dest_dir)):
        for _ in files:
            filecounter += 1
    return filecounter


def walkdir(folder):
    """Walk through each files in a directory"""
    for dirpath, _, files in os.walk(folder):
        for filename in files:
            yield os.path.abspath(os.path.join(dirpath, filename))


def plt_show(image_path):
    # 指定图片路径
    # 打开图片并显示
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')  # 隐藏坐标轴
    plt.show()


def plt_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    return plt


def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(root, file))
    return image_files


def get_image_hash(image_path):
    with Image.open(image_path) as img:
        img_hash = hashlib.md5(img.tobytes())
    return img_hash.hexdigest()


def find_duplicate_images(directory):
    image_files = get_image_files(directory)
    image_hashes = defaultdict(list)

    for image_file in image_files:
        image_hash = get_image_hash(image_file)
        image_hashes[image_hash].append(image_file)

    duplicate_images = {k: v for k, v in image_hashes.items() if len(v) > 1}
    return duplicate_images


def check_train_dir(dir_path):
    #     检查训练集目录，无效文件删除
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            file_path = os.path.join(root, file)
            print(file_path)
            os.remove(file_path) 

def check_dir_exists(dir_path):
    # 检查目录是否存在，不存在则创建
    if not os.path.exists(dir_path):
        print(f"创建目录 {dir_path}")
        os.makedirs(dir_path)


def copy_random_images(src_dir, train_dir, validation_dir):
    # 随机移动某些图片

    check_dir_exists(train_dir)
    check_dir_exists(validation_dir)
    all_images = []
    for root, _, files in os.walk(src_dir):
        # print(root,files)
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))

    num_images = int(len(all_images) * 0.1)
    if len(all_images) < num_images:
        raise ValueError(
            f"Source directory has only {len(all_images)} images, cannot move {num_images} images.")

    validation_images = random.sample(all_images, num_images)
    train_images = [img for img in all_images if img not in validation_images]

    # 遍历目标文件夹，如果不在目标list，则删除
    for root, _, files in os.walk(validation_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path not in validation_images:
                os.remove(file_path)
                print(f"Removed validation_dir {file_path}")

    for root, _, files in os.walk(train_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path not in validation_images:
                os.remove(file_path)
                print(f"Removed train_dir {file_path}")

    for img_path in validation_images:
        print(img_path)
        dest_path = os.path.join(validation_dir, os.path.basename(img_path))
        shutil.copyfile(img_path, dest_path)

    for img_path in train_images:
        dest_path = os.path.join(train_dir, os.path.basename(img_path))
        shutil.copyfile(img_path, dest_path)


def get_class_name(train_generator):
    # 获取分类标签
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    return class_names


def get_true_result(dest_dir, csv_path):
    # 获取真实结果
    data_frame = pd.read_csv(csv_path)
    # 获取真实结果
    for root, _, files in os.walk(dest_dir):
        for file in files:
            img_path = os.path.join(root, file)
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_name = os.path.basename(img_path)
            # 获取上一级目录名
            class_name = os.path.basename(root)
            # 根据img_name获取data_frame中的行 并更新值
            data_frame.loc[data_frame["img_name"] ==
                           img_name, "true_class"] = class_name
    data_frame.to_csv(csv_path, index=False, encoding="utf-8")
