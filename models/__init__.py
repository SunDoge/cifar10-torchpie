from .resnet import cifar_resnet20
import torch


def get_model(arch: str, pretrained=None):
    if arch == 'resnet20':
        model = cifar_resnet20(pretrained=pretrained)
        model = torch.jit.trace(model, torch.rand(1, 3, 32, 32))
        # print(model.graph)
        return model
    else:
        raise Exception(f'No such arch: {arch}')
