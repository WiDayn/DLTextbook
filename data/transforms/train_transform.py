import torchio


def train_transform():
    return torchio.Compose([
        torchio.RandomFlip(axes=(0, 1, 2)),  # 随机翻转
        torchio.RandomAffine(scales=(0.9, 1.1), degrees=15, translation=(5, 5, 5)),  # 加入位移变换
        torchio.RandomNoise(mean=0.0, std=(0.05, 0.15)),  # 噪声强度范围更灵活
        torchio.RandomBiasField(coefficients=(0.1, 0.3)),  # 模拟MRI偏置场伪影
        torchio.RandomBlur(std=(0.5, 1.5)),  # 加入随机模糊，模拟不同清晰度
        torchio.RandomGamma(log_gamma=(0.7, 1.5)),  # 调整图像对比度
        torchio.ZNormalization(),  # 标准化
    ])