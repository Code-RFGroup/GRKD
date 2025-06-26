import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Reduce
try:
    from torchvision.models.utils import load_state_dict_from_url
except:
    from torch.hub import load_state_dict_from_url
from typing import Callable, Any, Optional, List
import torch.nn.functional as F
def relation_similarity_metric(teacher, student, batch_data):
    #image, label = batch_data
    image=batch_data
    # Forward pass
    t_feats = teacher.forward_features(image)
    s_feats = student.forward_features(image)
    #print(t_feats.shape)
    # Get activation before average pooling
    #t_feat = t_feats[1]
   # s_feat = s_feats[1]
    # Compute batch similarity
    return -1 * batch_similarity(t_feats, s_feats)

def batch_similarity(f_t, f_s):
    bsz=4
    # Reshape
    f_s = f_s.view(f_s.shape[0], -1)#
    #print(f_s.shape)
    f_t = f_t.view(f_t.shape[0], -1)#B C*H*W
    # Get batch-wise similarity matrix
    G_s = torch.mm(f_s, torch.t(f_s))
    G_s = F.normalize(G_s)
    G_t = torch.mm(f_t, torch.t(f_t))
    G_t = F.normalize(G_t)
    #print(G_s.shape,G_t.shape)
    # Produce L_2 distance
    G_diff = G_t - G_s
    return (G_diff * G_diff).view(-1, 1).sum() / (bsz * bsz)
 
def semantic_similarity_metric(teacher, student, batch_data):
    criterion = nn.CrossEntropyLoss() 
    image, label = batch_data 
    # Forward once.
    t_logits = teacher.forward(image)
    s_logits = student.forward(image)
    # Backward once.
    criterion(t_logits, label).backward()
    criterion(s_logits, label).backward()
    # Grad-cam of fc layer.
    t_grad_cam = teacher.fc.weight.grad
    s_grad_cam = student.fc.weight.grad
    # Compute channel-wise similarity
    return -1 * channel_similarity(t_grad_cam, s_grad_cam)

def channel_similarity(f_t, f_s):
    bsz, ch = f_s.shape[0], f_s.shape[1]
    # Reshape
    f_s = f_s.view(bsz, ch, -1)
    f_t = f_t.view(bsz, ch, -1)
    # Get channel-wise similarity matrix
    emd_s = torch.bmm(f_s, f_s.permute(0, 2, 1))
    emd_s = F.normalize(emd_s, dim=2)
    emd_t = torch.bmm(f_t, f_t.permute(0, 2, 1))
    emd_t = F.normalize(emd_t, dim=2)
    # Produce L_2 distance
    G_diff = emd_s - emd_t
    return (G_diff * G_diff).view(bsz, -1).sum() / (ch * bsz)

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )
class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1
        # self.fc=nn.Sequential(Reduce('b c h w  -> b c', 'mean'),
        #             nn.Linear(oup, 22, bias=False)
        #         )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            x=x+self.conv(x)
            #return x + self.conv(x)
        else:
            x=self.conv(x)
            #return self.conv(x)
        #res=self.fc(x)
        #print(res.shape)
        return x


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 22,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        get_feat: bool = False,
        one=False,
        down=None
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))
        if one==True:
            features=nn.ModuleList([])
            features.append(Reduce('b c (h h1) (w w1) -> b c h w', 'mean',h1=down,w1=down))
            inp,oup=inverted_residual_setting[0][1],inverted_residual_setting[0][1]
            features.append(conv_1x1_bn(3,inp))
            for t,c,n,s in inverted_residual_setting:
                for i in range(n):
                    stride=1
                    features.append(block(inp,oup,stride,expand_ratio=t,norm_layer=norm_layer))
            features.append(conv_1x1_bn(oup,1280))
            self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        else:
            features=nn.ModuleList([])
            # building first layer
            input_channel = _make_divisible(input_channel * width_mult, round_nearest)
            self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
            features.append(ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer))
            # building inverted residual blocks
            for t, c, n, s in inverted_residual_setting:#17层
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                    input_channel = output_channel
            # building last several layers
            features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = features

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.get_feat = get_feat

    def _forward_impl(self, x: Tensor) -> Tensor:
        out=[]
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        # for i in range(len(self.features)):
        #     if i!=0 and i !=len
        for conv in self.features:
            if conv!=self.features[0] and conv !=self.features[-1]:
                x=conv(x)
                #out.append(output)
            else:
                x=conv(x)
            
        pre_f = x
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        f = nn.functional.adaptive_avg_pool2d(pre_f, (1, 1)).reshape(x.shape[0], -1)
        x = self.classifier(f)
        if self.get_feat == 'pre_GAP':
            return x, pre_f
        elif self.get_feat == 'after_GAP':
            return x, f
        else:
            return x
    def forward_features(self,x):
        for conv in self.features:
            if conv!=self.features[0] and conv !=self.features[-1]:
                x=conv(x)
                #out.append(output)
            else:
                x=conv(x)
        #pre_f=x
        #f = nn.functional.adaptive_avg_pool2d(pre_f, (1, 1)).reshape(x.shape[0], -1)
        return x
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def freeze_classifier(self):
        self.classifier.requires_grad = False



def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        path_checkpoint = "./mv2_best.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint,map_location='cuda:0')
        state_dict=checkpoint['net']
        model.load_state_dict(state_dict)
    return model

def get_blocks_to_drop():
    blocks = []
    blocks += ['features.3']
    blocks += ['features.5', 'features.6']
    blocks += ['features.8', 'features.9', 'features.10']
    blocks += ['features.12', 'features.13']
    blocks += ['features.15', 'features.16']
    return blocks
def mobilenet_v2_one_blocks(rm_blocks,pretrained=False,**kwargs):
    one_blocks= [0,0,0,0,0,0,0]
    inp,oup=0,0
    down=0
    for block in rm_blocks:
        if block in ['features.3']:
            one_blocks[1] += 1
            inp,oup=24,24
            down=2
        elif block in ['features.5', 'features.6']:#feature 4
            one_blocks[2] += 1
            inp,oup=32,32
            down=4
        elif block in ['features.8', 'features.9', 'features.10']:
            one_blocks[3] += 1
            inp,oup=64,64
            down=8

        elif block in ['features.12', 'features.13']:
            one_blocks[4] += 1
            inp,oup=96,96
            down=16
        elif block in ['features.15', 'features.16']:
            one_blocks[5] += 1
            inp,oup=160,160
            down=32
    inverted_residual_setting=[[6,inp,1,1]]
    model = MobileNetV2(inverted_residual_setting=inverted_residual_setting,one=True,down=down, **kwargs)
    if pretrained:
        path_checkpoint = "./mv2_best.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint,map_location='cuda:0')
        state_dict=checkpoint['net']
        load_one_block_state_dict(model, state_dict, rm_blocks)
    return model

def mobilenet_v2_rm_blocks(rm_blocks: list = [], pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    # rm_blocks = [[], [1], [1], [1], [1], [1], []]
    """
    1: 24->24: features.3
    2: 32->32: features.5,features.6
    3: 64->64: features.8,features.9,features.10
    4: 96->96: features.12,features.13
    5: 160->160: features.15,features.16
    """
    raw_blocks = [1, 2, 3, 4, 3, 3, 1]
    for block in rm_blocks:
        if block in ['features.3']:
            raw_blocks[1] -= 1
        elif block in ['features.5', 'features.6']:
            raw_blocks[2] -= 1
        elif block in ['features.8', 'features.9', 'features.10']:
            raw_blocks[3] -= 1
        elif block in ['features.12', 'features.13']:
            raw_blocks[4] -= 1
        elif block in ['features.15', 'features.16']:
            raw_blocks[5] -= 1

    print('=> mobilenet_v2 rm_blocks: {}'.format(rm_blocks))
    inverted_residual_setting = [
            # t, c, n, s
            [1, 16, raw_blocks[0], 1],
            [6, 24, raw_blocks[1], 2],
            [6, 32, raw_blocks[2], 2],
            [6, 64, raw_blocks[3], 2],
            [6, 96, raw_blocks[4], 1],
            [6, 160, raw_blocks[5], 2],
            [6, 320, raw_blocks[6], 1],
        ]
    model = MobileNetV2(inverted_residual_setting=inverted_residual_setting, **kwargs)
    if pretrained:
        path_checkpoint = "./mv2_best.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint,map_location='cuda:0')
        state_dict=checkpoint['net']
        #net.load_state_dict(state_dict['net'])
        # state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
        #                                       progress=progress)
        load_rm_block_state_dict(model, state_dict, rm_blocks)
        
    return model
def load_one_block_state_dict(model, raw_state_dict, rm_blocks):#第二个参数是训练好的模型的dict
    state_dict = model.state_dict()
    #print(model)
    num=rm_blocks[0].split('.')[1]
    for raw_key in raw_state_dict.keys():
        key_items = raw_key.split('.')
        #print(key_items)
        if key_items[0]=='features' and key_items[1]==num:
            key_items[1]='2'
            target_key = '.'.join(key_items)
            #print(target_key)
           # print(state_dict)
            #print(target_key)
            assert target_key in state_dict
            state_dict[target_key] = raw_state_dict[raw_key]
        elif key_items[0]=='classifier':
            assert raw_key in state_dict
            #print(raw_key)
            state_dict[raw_key] = raw_state_dict[raw_key]
    model.load_state_dict(state_dict)
    pass
def load_rm_block_state_dict(model, raw_state_dict, rm_blocks):
    rm_count = 0
    has_count = set()
    state_dict = model.state_dict()
    for raw_key in raw_state_dict.keys():
        key_items = raw_key.split('.')
        if key_items[0] == 'features':
            block = f'features.{key_items[1]}'
            if block in rm_blocks:
                if block not in has_count:
                    has_count.add(block)
                    rm_count += 1
            else:
                key_items[1] = str(int(key_items[1]) - rm_count)
                target_key = '.'.join(key_items)
                assert target_key in state_dict
                state_dict[target_key] = raw_state_dict[raw_key]
        else:
            assert raw_key in state_dict
            state_dict[raw_key] = raw_state_dict[raw_key]
    model.load_state_dict(state_dict)


if __name__ == "__main__":
    x = torch.FloatTensor(4, 3, 256, 256)
    teacher_model = mobilenet_v2(pretrained=False)
    student_model=mobilenet_v2_rm_blocks(rm_blocks=['features.3'],pretrained=False)
    #model=mobilenet_v2_one_blocks(rm_blocks=['features.16'],pretrained=False)
    #model=mobilenet_v2_rm_blocks(rm_blocks=['features.3'],pretrained=False)
    
    # for n in model.named_modules():
    #     print(n)
    pred=teacher_model(x)
    print(pred.shape)
    pred=student_model(x)
    print(pred.shape)
    print(relation_similarity_metric(teacher_model, student_model, x))

    #print(len(pred))

    '''
    model = mobilenet_v2(pretrained=True)
    
    y = model(x)
    print(y.data.shape)
    from torchsummaryX import summary
    summary(model, torch.zeros((1, 3, 224, 224)))
    '''

    # rm_blocks = [[], [1], [1], [1], [1], [1], []]
    # rm_blocks = ['features.3', 'features.5', 'features.8', 'features.12', 'features.15']
    # prune_model = mobilenet_v2_rm_blocks(rm_blocks, pretrained=True)
    # y = prune_model(x)
    # import IPython
    # IPython.embed()
