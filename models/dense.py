from monai.networks.nets import DenseNet, DenseNet121


def DENSE():
    model = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
    )
    return model
