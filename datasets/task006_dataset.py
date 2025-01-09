import glob
import random

import numpy as np
import torch
import torchio as tio

import glob
import random
import os

class Task006Dataset:
    def __init__(self, root_dir, device, transform=None, resize=(64, 64, 64), set_type="train", fold=4, num_folds=5, sequence=""):
        self.set_type = set_type
        self.root_dir = root_dir
        self.transform = transform
        self.resize = resize
        # self.resize = None
        self.num_folds = num_folds  # 总折数，默认为5
        self.fold = fold  # 当前折数
        self.image_files = glob.glob(root_dir + '/**/US_src_ori.nii.gz', recursive=True)
        self.device = device

        # 随机打乱文件顺序
        random.seed(42)
        random.shuffle(self.image_files)

        # 计算每折包含的数据量
        fold_size = len(self.image_files) // self.num_folds

        # 计算当前fold的起始和结束索引
        val_start_idx = self.fold * fold_size
        val_end_idx = (self.fold + 1) * fold_size if self.fold != self.num_folds - 1 else len(self.image_files)

        # 按fold划分数据集
        val_files = self.image_files[val_start_idx:val_end_idx]
        train_files = self.image_files[:val_start_idx] + self.image_files[val_end_idx:]

        # 根据set_type决定使用哪个数据集
        if self.set_type == "train":
            self.image_files = train_files
        elif self.set_type == "val":
            self.image_files = val_files
        elif self.set_type == "test":
            pass
        else:
            raise ValueError("set_type must be either 'train' or 'val'")

        print(f"Fold {self.fold + 1}: {self.set_type.capitalize()} set has {len(self.image_files)} files.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_1 = tio.ScalarImage(self.image_files[idx])
        label = float(self.image_files[idx].split('/')[-3])

        # 数据增强
        if self.transform:
            subject_1 = tio.Subject(image=image_1)
            subject_1 = self.transform(subject_1)
            image_1 = subject_1.image

        # 如果需要resize
        if self.resize:
            resize_transform = tio.Resize(self.resize)
            image_1 = resize_transform(image_1)

            crop_or_pad = tio.CropOrPad(self.resize)
            image_1 = crop_or_pad(image_1)


        # 转换为numpy数组
        image_1 = image_1.data.numpy().astype(np.float32)

        return torch.tensor(image_1).to(self.device), torch.tensor(int(label)).to(self.device)
