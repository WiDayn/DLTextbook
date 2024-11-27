import torchio


def train_transform():
    return torchio.Compose([
    torchio.RandomFlip(axes=(0, 1, 2)),
    torchio.RandomAffine(scales=(0.9, 1.1), degrees=15),
    torchio.RandomNoise(mean=0.0, std=0.1),
    torchio.ZNormalization(),
    # torchio.RescaleIntensity((-1, 1)),
    # 可以根据需要添加更多的数据增强操作
])