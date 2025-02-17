import importlib
from torchvision import models

from .lenet import LeNet
from .vggnet import VggNet
from .resnet32 import resnet32
from .slimresnet18 import slimresnet18, resnet18_cifar
from .resnet18_imagenet_lucir import resnet18_imagenet_lucir
from .resnet18_lucir import resnet18_lucir, slimresnet18_lucir
from .resnet_18_ssre import resnet18_ssre, slimresnet18_ssre
from .resnet_cbam import resnet18_cbam
from .resnet_wa import resnet18_wa_cbam
from .resnet_18_ssre_with_bn import resnet18_ssre_bn, slimresnet18_ssre_bn

from .resnet18_tiny_imagenet import resnet18_tiny_imagenet, slimresnet18_tiny_imagenet
from .resnet_18_ssre_tiny_imagenet import resnet18_ssre_bn_tiny

# available torchvision models
tvmodels = ['alexnet',
            'densenet121', 'densenet169', 'densenet201', 'densenet161',
            'googlenet',
            'inception_v3',
            'mobilenet_v2',
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
            'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
            'squeezenet1_0', 'squeezenet1_1',
            'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
            'wide_resnet50_2', 'wide_resnet101_2'
            ]

allmodels = tvmodels + ['resnet32', 'LeNet', 'VggNet', 'slimresnet18', 'resnet18_cifar',
                        "resnet18_lucir", "resnet18_imagenet_lucir",
                        "resnet18_ssre", "slimresnet18_ssre",
                        "resnet18_cbam", "resnet18_wa_cbam",
                        "resnet18_ssre_bn", "slimresnet18_ssre_bn",
                        "resnet18_tiny_imagenet", "slimresnet18_tiny_imagenet", "resnet18_ssre_bn_tiny"]

def set_tvmodel_head_var(model):
    if type(model) == models.AlexNet:
        model.head_var = 'classifier'
    elif type(model) == models.DenseNet:
        model.head_var = 'classifier'
    elif type(model) == models.Inception3:
        model.head_var = 'fc'
    elif type(model) == models.ResNet:
        model.head_var = 'fc'
    elif type(model) == models.VGG:
        model.head_var = 'classifier'
    elif type(model) == models.GoogLeNet:
        model.head_var = 'fc'
    elif type(model) == models.MobileNetV2:
        model.head_var = 'classifier'
    elif type(model) == models.ShuffleNetV2:
        model.head_var = 'fc'
    elif type(model) == models.SqueezeNet:
        model.head_var = 'classifier'
    else:
        raise ModuleNotFoundError

def get_base_network(network_name, pretrained=False):
    from networks.network import LLL_Net
    if network_name in tvmodels:  # torchvision models
        tvnet = getattr(importlib.import_module(name='torchvision.models'), network_name)
        if network_name == 'googlenet':
            init_model = tvnet(pretrained=pretrained, aux_logits=False)
        else:
            init_model = tvnet(pretrained=pretrained)
        set_tvmodel_head_var(init_model)
    else:  # other models declared in networks package's init
        net = getattr(importlib.import_module(name='networks'), network_name)
        # WARNING: fixed to pretrained False for other model (non-torchvision)
        init_model = net(pretrained=False)
    return init_model
