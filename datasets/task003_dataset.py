import glob
import random

import numpy as np
import pandas as pd
import torch
import torchio as tio

import glob
import random
import os

from scipy.stats import stats


class Task003Dataset:
    def __init__(self, root_dir, device, transform=None, resize=(64, 64, 64), set_type="train", fold=4, num_folds=5, sequence=""):
        self.set_type = set_type
        self.root_dir = root_dir
        self.transform = transform
        self.resize = resize
        # self.resize = None
        self.num_folds = num_folds  # 总折数，默认为5
        self.fold = fold  # 当前折数
        # self.image_files = glob.glob(r"/home/jjf/PycharmProjects/swinUMamba/DLTextbook/datasets/raw/rxa_test/**/"+sequence+"-5_src_ori.nii.gz", recursive=True)
        self.image_files = glob.glob(os.path.join(root_dir, '**/'+sequence+'_image_ROI_5_5_5.nii.gz'), recursive=True)
        if len(self.image_files) == 0:
            self.image_files = glob.glob(os.path.join(root_dir, '**/'+sequence+'-5_src_ori.nii.gz'), recursive=True)
        # self.genomics_file = "./TrainSelectFeature.csv"
        self.genomics_train_file ="./datasets/raw/rxa_rad/TrainSelectFeature_"+sequence+".csv"
        self.genomics_val_file = "./datasets/raw/rxa_rad/InternalSelectFeature_"+sequence+".csv"
        self.genomics_test_file = "./datasets/raw/rxa_rad/ExternalSelectFeature_"+sequence+".csv"
        #
        train_df = pd.read_csv(self.genomics_train_file)
        train_df = train_df.set_index("StudyID").drop(columns=["ClassifyValue"])
        val_df = pd.read_csv(self.genomics_val_file)
        val_df = val_df.set_index("StudyID").drop(columns=["ClassifyValue"])
        test_df = pd.read_csv(self.genomics_test_file)
        test_df = test_df.set_index("StudyID").drop(columns=["ClassifyValue"])
        # common_columns = test_df.columns.intersection(df.columns)
        # test_df = test_df[common_columns]
        # train_df = train_df[common_columns]
        combined_series = pd.concat([train_df, val_df, test_df])

        # for column in train_df.columns:
        #     mean = train_df[column].mean()
        #     std = train_df[column].std()
        #
        #     train_df[column] = (train_df[column] - mean) / std
        #
        #     test_df[column] = (test_df[column] - mean) / std
        #
        #     combined_series[column] = (combined_series[column] - mean) / std
        # 将每行转换为张量，并构建字典
        self.data_dict = {idx: torch.tensor(row.values, dtype=torch.float32) for idx, row in combined_series.iterrows()}
        self.device = device

        # 随机打乱文件顺序
        random.seed(42)
        random.shuffle(self.image_files)

        if fold == -1:
            print(f"{set_type}: {self.set_type.capitalize()} set has {len(self.image_files)} files.")
            return

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
            raise ValueError("set_type must be either 'train' or 'val' or 'test'")

        print(f"Fold {self.fold + 1}: {self.set_type.capitalize()} set has {len(self.image_files)} files.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_1 = tio.ScalarImage(self.image_files[idx])
        # image_2 = tio.ScalarImage(self.image_files[idx].replace("V40", "VZ"))
        label = float(self.image_files[idx].split('/')[-3])
        name = self.image_files[idx].split('/')[-2]

        # 数据增强
        if self.transform:
            subject_1 = tio.Subject(image=image_1)
            subject_1 = self.transform(subject_1)
            image_1 = subject_1.image

            # subject_2 = tio.Subject(image=image_2)
            # subject_2 = self.transform(subject_2)
            # image_2 = subject_2.image

        # 如果需要resize
        if self.resize:
            resize_transform = tio.Resize(self.resize)
            image_1 = resize_transform(image_1)
            # image_2 = resize_transform(image_2)

            crop_or_pad = tio.CropOrPad(self.resize)
            image_1 = crop_or_pad(image_1)
            # image_2 = crop_or_pad(image_2)

        # 转换为numpy数组
        image_1 = image_1.data.numpy().astype(np.float32)
        # image_2 = image_2.data.numpy().astype(np.float32)

        # 合并为两个通道的图像
        # merged_image = np.stack((image_1, image_2), axis=0).squeeze()  # 现在是形状 (2, H, W)

        omics_features_expanded = self.data_dict[name].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [4, 10, 1, 1, 1]
        omics_features_expanded = omics_features_expanded.expand(len(self.data_dict[name]), *self.resize)  # [4, 10, 64, 128, 128]

        merged_features = torch.cat([torch.tensor(image_1), omics_features_expanded], dim=0).to(self.device)

        return merged_features.to(self.device), torch.tensor(int(label)).to(self.device)


