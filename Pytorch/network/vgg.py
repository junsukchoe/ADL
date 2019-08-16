import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class ADL(nn.Module):
    def __init__(self, drop_rate=0.75, drop_thr=0.8):
        super(ADL, self).__init__()
        assert 0 <= drop_rate <= 1 and 0 <= drop_thr <= 1
        self.drop_rate = drop_rate
        self.drop_thr = drop_thr

        self.attention = None
        self.drop_mask = None

    def extra_repr(self):
        return 'drop_rate={}, drop_thr={}'.format(
            self.drop_rate, self.drop_thr
        )

    def forward(self, x):
        if self.training:
            b = x.size(0)

            # Generate self-attention map
            attention = torch.mean(x, dim=1, keepdim=True)
            self.attention = attention
            # Generate importance map
            importance_map = torch.sigmoid(attention)

            # Generate drop mask
            max_val, _ = torch.max(attention.view(b, -1), dim=1, keepdim=True)
            thr_val = max_val * self.drop_thr
            thr_val = thr_val.view(b, 1, 1, 1).expand_as(attention)
            drop_mask = (attention < thr_val).float()
            self.drop_mask = drop_mask
            # Random selection
            random_tensor = torch.rand([], dtype=torch.float32) + self.drop_rate
            binary_tensor = random_tensor.floor()
            selected_map = (1. - binary_tensor) * importance_map + binary_tensor * drop_mask

            # Spatial multiplication to input feature map
            output = x.mul(selected_map)

        else:
            output = x
        return output

    def get_maps(self):
        return self.attention, self.drop_mask


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, mode='base', init_weights=True, **kwargs):
        super(VGG, self).__init__()
        self.features = features
        self.mode = mode

        if self.mode == 'base':
            self.avgpool = nn.AdaptiveAvgPool2d((7,7))
            self.classifier = make_classifier(mode = self.mode, num_classes=num_classes)

        elif self.mode == 'GAP' or self.mode == 'ADL':
            self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            if self.mode == 'ADL' and kwargs['lst_ADL']:
                self.lst_adl = True
                self.adl = ADL(kwargs['drop_rate'], kwargs['drop_thr'])
            else:
                self.lst_adl = False
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(1024, num_classes)

        self.feature_map = None
        self.pred = None

        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        x = self.features(x)
        if self.mode == 'base':
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        elif self.mode == 'GAP' or self.mode == 'ADL':
            x = self.conv6(x)
            x = self.relu(x)
            if self.lst_adl:
                x = self.adl(x)
            self.feature_map = x
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            self.pred = x

        else:
            raise Exception("No mode matching")

        return x

    def get_cam(self):
        return self.feature_map, self.pred


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, **kwargs):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [ADL(kwargs['drop_rate'], kwargs['drop_thr'])]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_classifier(mode, num_classes):
    if mode == 'base':
        return nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    elif mode == 'GAP':
        pass
    else:
        raise Exception("No mode matching")


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'D_GAP': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
}

def make_cfg(cfg, targets):
    pos_list = list('0')
    block, element = 1, 1
    for i in cfg:
        if isinstance(i, int):
            pos_list.append(str(block) + str(element))
            element += 1
        elif isinstance(i, str):
            pos_list.append(str(block) + 'M')
            block += 1
            element = 1
    pos_dict = dict()
    for i, key in enumerate(pos_list):
        pos_dict[key] = i
    target_list = [pos_dict[key] for key in targets]
    target_list = sorted(target_list, reverse=True)

    for key in target_list:
        cfg.insert(key, 'A')
    return cfg

def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict


def make_config_align(state_dict, base_cfg, target_cfg):
    base_list = []
    replace_list = []
    start = 0
    for layer in base_cfg:
        if type(layer) is int:
            base_list.append(start)
            start += 2
        elif type(layer) is str:
            start += 1
        else:
            raise Exception("No state_dict match")

    start = 0
    for layer in target_cfg:
        if type(layer) is int:
            replace_list.append(start)
            start += 2
        elif type(layer) is str:
            start += 1
        else:
            raise Exception("No state_dict match")
    target_list = {}
    for base, replace in zip(base_list, replace_list):
        target_list[str(base)] = str(replace)
    keys = [key for key in state_dict.keys()]
    for i, key in enumerate(reversed(keys)):
        key_ = key.split('.')
        new_key = key.replace('.'+key_[1]+'.', '.'+target_list[key_[1]]+'.')
        state_dict[new_key] = state_dict.pop(key)

    return state_dict


def _vgg(arch, cfg, mode, batch_norm, pretrained, progress, **kwargs):
    kwargs['init_weights'] = True
    if mode == 'ADL':
        if '6' in kwargs['ADL_position']:
            kwargs['lst_ADL'] = True
            kwargs['ADL_position'].remove('6')
        else:
            kwargs['lst_ADL'] = False
        new_cfg = make_cfg(cfgs[cfg], kwargs['ADL_position'])
    else:
        new_cfg = cfgs[cfg]
    model = VGG(make_layers(new_cfg, batch_norm=batch_norm, **kwargs), mode=mode, **kwargs)

    if pretrained:
        state_dict = load_url(model_urls[arch], progress=progress)

        # remove classifier part
        if mode == 'ADL':
            state_dict = remove_layer(state_dict, 'classifier.')
            make_config_align(state_dict, cfgs[cfg[0]], cfgs[cfg])
            strict_rule = False
        elif mode == 'GAP':
            state_dict = remove_layer(state_dict, 'classifier.')
            strict_rule = False
        else:
            strict_rule = True
            # If not ImageNet, remove pretrained dict.
            if kwargs['num_classes'] != 1000:
                remove_layer(state_dict, 'classifier.6')
                strict_rule = False

        model.load_state_dict(state_dict, strict=strict_rule)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', 'base', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', 'base', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', 'base', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', 'base', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', 'base', False, pretrained, progress, **kwargs)


def vgg16_ADL(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D_GAP', 'ADL', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', 'base', True, pretrained, progress, **kwargs)

def vgg16_GAP(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D_GAP', 'GAP', False, pretrained, progress, **kwargs)

def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', 'base', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', 'base', True, pretrained, progress, **kwargs)