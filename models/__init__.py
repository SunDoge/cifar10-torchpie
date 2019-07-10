from .resnet import cifar_resnet20


def get_model(arch: str, pretrained=None):
    if arch == 'resnet20':
        return cifar_resnet20(pretrained=pretrained)
    else:
        raise Exception(f'No such arch: {arch}')
