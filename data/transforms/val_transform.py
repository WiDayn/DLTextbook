import torchio


def val_transform():
    return torchio.Compose([
        torchio.ZNormalization(),
        # torchio.RescaleIntensity((-1, 1)),
        # 可以根据需要添加更多的数据增强操作
])