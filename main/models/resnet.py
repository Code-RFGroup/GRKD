import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Reduce
try:
    from torchvision.models.utils import load_state_dict_from_url
except:
    from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 22,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        get_feat: str = 'None',
        one=False,
        down=None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.one=one
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.layer1=None
        self.layer2=None
        self.layer3=None
        self.layer4=None
        if one==True:
            #self.conv=conv1x1(3,64)
            if layers[0]!=0:
                self.reduce=Reduce('b c (h h1) (w w1) -> b c h w', 'mean',h1=4,w1=4)
                self.conv=conv_1x1_bn(3,256)
                self.layer1=self._make_layer(block, 64, layers[0],one=True)
                self.conv2=conv_1x1_bn(256,2048)
            elif layers[1]!=0:
                self.reduce=Reduce('b c (h h1) (w w1) -> b c h w', 'mean',h1=8,w1=8)
                self.conv=conv_1x1_bn(3,512)
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],one=True)
                self.conv2=conv_1x1_bn(512,2048)
            elif layers[2]!=0:
                self.reduce=Reduce('b c (h h1) (w w1) -> b c h w', 'mean',h1=16,w1=16)
                self.conv=conv_1x1_bn(3,1024)
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],one=True)
                self.conv2=conv_1x1_bn(1024,2048)                       
            elif layers[3]!=0:
                self.reduce=Reduce('b c (h h1) (w w1) -> b c h w', 'mean',h1=32,w1=32)
                self.conv=conv_1x1_bn(3,2048)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],one=True)
                self.conv2=conv_1x1_bn(2048,2048)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
                    
        self.get_feat = get_feat

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False,one=False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        layers = []
        if one == True:
            self.inplanes = planes * block.expansion
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        else:
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )


            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)



    def _forward_impl(self, x: Tensor) -> Tensor:
        if self.one:
            x=self.reduce(x)
            x=self.conv(x)
            if self.layer1:
                x=self.layer1(x)
            elif self.layer2:
                x=self.layer2(x)
            elif self.layer3:
                x=self.layer3(x)
            elif self.layer4:
                x=self.layer4(x)
            pre_f=self.conv2(x)
        else:
        # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            #print('layer0',x.shape)
            x = self.layer1(x)
            #print('layer1',x.shape)
            x = self.layer2(x)
            #print('layer2',x.shape)
            x = self.layer3(x)
            #print('layer3',x.shape)
            pre_f = self.layer4(x)
            #print('layer4',pre_f.shape)
        #print(pre_f.shape)
        
        x = self.avgpool(pre_f)
        f = torch.flatten(x, 1)
        x = self.fc(f)
        if self.get_feat == 'pre_GAP':
            return x, pre_f
        elif self.get_feat == 'after_GAP':
            return x, f
        else:
            return x
    def forward_features(self,x):
        if self.one:
            x=self.reduce(x)
            x=self.conv(x)
            if self.layer1:
                x=self.layer1(x)
            elif self.layer2:
                x=self.layer2(x)
            elif self.layer3:
                x=self.layer3(x)
            elif self.layer4:
                x=self.layer4(x)
            pre_f=self.conv2(x)
        else:
        # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            #print('layer0',x.shape)
            x = self.layer1(x)
            #print('layer1',x.shape)
            x = self.layer2(x)
            #print('layer2',x.shape)
            x = self.layer3(x)
            #print('layer3',x.shape)
            pre_f = self.layer4(x)
        return pre_f
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def freeze_classifier(self):
        self.fc.requires_grad = False





def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        if arch=='resnet50':
            path_checkpoint = "./resnet_best.pth"
        else:
            path_checkpoint = "./resnet34.pth"
        checkpoint = torch.load(path_checkpoint,map_location='cuda:0')
        state_dict=checkpoint['net']
        fix_state_dict(model.state_dict(), state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model

def fix_state_dict(target_state_dict, state_dict):
    for key in target_state_dict:
        if key not in state_dict.keys():
            print('=> not find {}'.format(key))
            continue
        if target_state_dict[key].shape != state_dict[key].shape:
            state_dict.pop(key)
            print('=> pop {}'.format(key))
    return state_dict


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet34_rm_blocks(rm_blocks: list = [], pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    #rm_blocks = ['layer1.1', 'layer2.1', 'layer3.1', 'layer4.1']
    print('remove blocks {}'.format(rm_blocks))
    blocks = [3, 4, 6, 3]
    for block in rm_blocks:
        layer_num = int(block[5]) - 1
        blocks[layer_num] -= 1
    model = ResNet(BasicBlock, blocks, **kwargs)
    if pretrained:
        path_checkpoint = "./resnet34.pth"
        checkpoint = torch.load(path_checkpoint,map_location='cuda:0')
        state_dict=checkpoint['net']
        load_rm_block_state_dict(model, state_dict, rm_blocks)
        
    return model



def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def load_rm_block_state_dict(model, raw_state_dict, rm_blocks, verbose=False):
    state_dict = model.state_dict()
    layer_rm_count = {
        'layer1': 0,
        'layer2': 0,
        'layer3': 0,
        'layer4': 0,
    }
    has_count = set()
    state_dict = model.state_dict()
    target_keys = set(state_dict.keys())
    for raw_key in raw_state_dict.keys():
        if 'num_batches_tracked' in raw_key:
            if verbose:
                print(f'not load {raw_key}')
            continue
        if 'downsample' in raw_key and int(raw_key.split('.')[1]) > 0:
            if verbose:
                print(f'not load {raw_key}')
            continue
        items = raw_key.split('.')
        layer, block = items[0], items[1]
        if 'layer' not in layer:
            state_dict[raw_key] = raw_state_dict[raw_key]
            target_keys.discard(raw_key)
            if verbose:
                print(f'T:{raw_key}\t -> S:{raw_key}')
            continue
        check_rm = "{}.{}".format(layer, block)
        if check_rm in rm_blocks:
            if check_rm not in has_count:
                has_count.add(check_rm)
                layer_rm_count[layer] += 1
        else:
            items[1] = str(int(block) - layer_rm_count[layer])
            target_key = '.'.join(items)
            assert target_key in state_dict
            state_dict[target_key] = raw_state_dict[raw_key]
            target_keys.discard(target_key)
            if verbose:
                print(f'T:{raw_key} -> S:{target_key}')
    model.load_state_dict(state_dict)
    if verbose:
        print(f'student has not loaded {target_keys}')


def resnet50_rm_blocks(rm_blocks: list = [], pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    #rm_blocks = ['layer1.1', 'layer2.1', 'layer3.1', 'layer3.4', 'layer4.1']
    #rm_blocks = ['layer1.1', 'layer2.1', 'layer3.1', 'layer4.1']
    print('remove blocks {}'.format(rm_blocks))
    blocks = [3, 4, 6, 3]
    for block in rm_blocks:
        layer_num = int(block[5]) - 1
        blocks[layer_num] -= 1
    model = ResNet(Bottleneck, blocks, **kwargs)
    if pretrained:
        path_checkpoint = "./resnet_best.pth"
        checkpoint = torch.load(path_checkpoint,map_location='cuda:0')
        state_dict=checkpoint['net']
        load_rm_block_state_dict(model, state_dict, rm_blocks)
    return model
def resnet50_one_blocks(rm_blocks,pretrained=False,**kwargs):
    blocks = [0, 0, 0, 0]
    for block in rm_blocks:
        layer_num = int(block[5]) - 1
        blocks[layer_num] += 1
    model = ResNet(Bottleneck, blocks,one=True, **kwargs)
    if pretrained:
        path_checkpoint = "./resnet_best.pth"
        checkpoint = torch.load(path_checkpoint,map_location='cuda:0')
        state_dict=checkpoint['net']
        #state_dict = torch.load(path_checkpoint,map_location='cuda:0')
        load_one_block_state_dict(model, state_dict, rm_blocks)
        print('load one model')
    return model

def load_one_block_state_dict(model, raw_state_dict, rm_blocks):#第二个参数是训练好的模型的dict
    state_dict = model.state_dict()

    num1,num2=rm_blocks[0].split('.')[0],rm_blocks[0].split('.')[1]

    for raw_key in raw_state_dict.keys():
        items = raw_key.split('.')
        layer, block = items[0], items[1]#'#layer2.1'
        if 'layer' in layer and layer==num1 and block==num2: 
            items[1]='0'
            #print('1')
            target_key='.'.join(items)
            state_dict[target_key] = raw_state_dict[raw_key]
        elif items=='fc':
            state_dict[raw_key] = raw_state_dict[raw_key]
    model.load_state_dict(state_dict)

def get_blocks_to_drop(model):
    blocks = []
    for layer_id in range(1, 5):
        layer = getattr(model, f'layer{layer_id}')
        for block_id in range(1, len(layer)):
            block_name = f'layer{layer_id}.{block_id}'
            blocks.append(block_name)
    return blocks

if __name__ == "__main__":
    model=resnet50_one_blocks(rm_blocks=['layer3.5'],pretrained=True)
    
    #model = resnet50(pretrained=False)
    x = torch.FloatTensor(1, 3, 256, 256)
    y = model(x)
    # for n in model.named_modules():
    #     print(n)
    # for para in model.parameters():
    #     print(para)
    #print(get_blocks_to_drop(model))
    print(y.data.shape)
    # from torchsummaryX import summary
    # summary(model, torch.zeros((1, 3, 224, 224)))
    #import IPython
    #IPython.embed()
