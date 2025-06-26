import torch.nn as nn
    # elif 'repvit' in model_name:
    #     t_grad_cam = teacher.classifier.classifier.l.weight.grad
    #     s_grad_cam = student.classifier.classifier.l.weight.grad

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

from timm.models.layers import SqueezeExcite
import torch.nn.functional as F
import torch
from torch import nn
from torch import Tensor
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
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    
    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert(m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
    
    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

from timm.models.vision_transformer import trunc_normal_
class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self):
        classifier = self.classifier.fuse()
        if self.distillation:
            classifier_dist = self.classifier_dist.fuse()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier

class RepViT(nn.Module):
    def __init__(self, cfgs, num_classes=1000, distillation=False,get_feat=False):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.get_feat=get_feat
        # building first layer
        input_channel = 40
        patch_embed = torch.nn.Sequential(Conv2d_BN(3, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                           Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)
        self.classifier = Classfier(output_channel, num_classes, distillation)
    def forward_features(self,x):
        for f in self.features:
            x = f(x)
        return x
    def forward(self, x):
        # x = self.features(x)
        for f in self.features:
            x = f(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.classifier(x)
        if self.get_feat == 'pre_GAP':
            return x,None
        elif self.get_feat == 'after_GAP':
            return x,None
        # else:
        #     return [x]+out
        return x
        #return x

from timm.models import register_model


#@register_model
def repvit_m0_6(cfgs=None,pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    if cfgs is None:
        cfgs = [
            [3,   2,  40, 1, 0, 1],
            [3,   2,  40, 1, 0, 1],
            [3,   2,  80, 0, 0, 2],#3
            [3,   2,  80, 1, 0, 1],
            [3,   2,  80, 1, 0, 1],
            [3,   2,  160, 0, 1, 2],#6
            [3,   2, 160, 1, 1, 1],
            [3,   2, 160, 1, 1, 1],
            [3,   2, 160, 1, 1, 1],
            [3,   2, 160, 1, 1, 1],
            [3,   2, 160, 1, 1, 1],
            [3,   2, 160, 1, 1, 1],
            [3,   2, 160, 1, 1, 1],
            [3,   2, 160, 1, 1, 1],
            [3,   2, 160, 1, 1, 1],
            [3,   2, 320, 0, 1, 2],#16
            [3,   2, 320, 1, 1, 1],
        ]
    #     cfgs = [
    #     [3,   2,  40, 1, 0, 1],
    #     [3,   2,  40, 0, 0, 1],
    #     [3,   2,  80, 0, 0, 2],#3
    #     [3,   2,  80, 1, 0, 1],
    #     [3,   2,  80, 0, 0, 1],
    #     [3,   2,  160, 0, 1, 2],#6
    #     [3,   2, 160, 1, 1, 1],
    #     [3,   2, 160, 0, 1, 1],
    #     [3,   2, 160, 1, 1, 1],
    #     [3,   2, 160, 0, 1, 1],
    #     [3,   2, 160, 1, 1, 1],
    #     [3,   2, 160, 0, 1, 1],
    #     [3,   2, 160, 1, 1, 1],
    #     [3,   2, 160, 0, 1, 1],
    #     [3,   2, 160, 0, 1, 1],
    #     [3,   2, 320, 0, 1, 2],#16
    #     [3,   2, 320, 1, 1, 1],
    # ]
    model=RepViT(cfgs, num_classes=num_classes, distillation=distillation)
    if pretrained:
        print("load...teacher:")
        path_checkpoint = "./repvit_best.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint,map_location='cuda:0')
        state_dict=checkpoint['net']
        model.load_state_dict(state_dict)
        # state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
        #                                       progress=progress)
        #load_rm_block_state_dict(model, state_dict, rm_blocks)

   # print("no load teacher")
    return model
    #return RepViT(cfgs, num_classes=num_classes, distillation=distillation)

def get_blocks_to_drop():
    blocks = []
    blocks += ['features.1','features.2']
    blocks += ['features.4', 'features.5']
    blocks += ['features.7', 'features.8', 'features.9','features.10', 'features.11', 'features.12', 'features.13', 'features.14', 'features.15']
    blocks += ['features.17']
    return blocks
def repvit_rm_blocks(rm_blocks: list = [], pretrained: bool = False,num_classes=1000,  **kwargs):
    # rm_blocks = [[], [1], [1], [1], [1], [1], []]
    """
    1: 24->24: features.3
    2: 32->32: features.5,features.6
    3: 64->64: features.8,features.9,features.10
    4: 96->96: features.12,features.13
    5: 160->160: features.15,features.16
    """
    raw_blocks = [1,2,1,2,1,9,1,1]
    for block in rm_blocks:
        if block in ['features.1','features.2']:
            raw_blocks[1] -= 1
        elif block in ['features.4', 'features.5']:
            raw_blocks[3] -= 1
        elif block in ['features.7', 'features.8', 'features.9','features.10', 'features.11', 'features.12', 'features.13', 'features.14', 'features.15']:
            raw_blocks[5] -= 1
        elif block in ['features.17']:
            raw_blocks[7] -= 1

    print('=> repvit rm_blocks: {}'.format(rm_blocks))
    cfgs=[]
    #print(num_classes)
    for i in range(raw_blocks[1]):
        cfgs.append([3,   2,  40, 1, 0, 1])
    cfgs.append([3,   2,  80, 0, 0, 2])
    for i in range(raw_blocks[3]):
        cfgs.append([3,   2,  80, 1, 0, 1])
    cfgs.append([3,   2,  160, 0, 1, 2])
    for i in range(raw_blocks[5]):
        cfgs.append([3,   2, 160, 1, 1, 1])
    cfgs.append([3,   2, 320, 0, 1, 2])
    for i in range(raw_blocks[7]):
        cfgs.append([3,   2, 320, 1, 1, 1])

    model =repvit_m0_6(cfgs=cfgs,num_classes=num_classes)
    if pretrained:
       # print("load...teacher:")
        path_checkpoint = "./repvit_best.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint,map_location='cuda:0')
        state_dict=checkpoint['net']
        #net.load_state_dict(state_dict['net'])
        # state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
        #                                       progress=progress)
        load_rm_block_state_dict(model, state_dict, rm_blocks)

   # print("no load teacher")
    return model
def load_rm_block_state_dict(model, raw_state_dict, rm_blocks):
    rm_count = 0
    has_count = set()
    state_dict = model.state_dict()
    # for k in state_dict.keys():
    #     print(k)
    # for k in raw_state_dict.keys():
    #     print(k)
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
                #print(target_key,raw_key,rm_count)
                assert target_key in state_dict
                state_dict[target_key] = raw_state_dict[raw_key]
        else:
            assert raw_key in state_dict
            #print(raw_key)
            state_dict[raw_key] = raw_state_dict[raw_key]
    print('load_checkpoint')       
    model.load_state_dict(state_dict)


from ptflops import get_model_complexity_info

if __name__ == "__main__":
    rm_blocks=get_blocks_to_drop()
    model=repvit_m0_6(num_classes=22)
    #s_model=repvit_rm_blocks(rm_blocks=rm_blocks,num_classes=22)
    img=torch.FloatTensor(4, 3, 256, 256)#0，3，6，16下采样
    # pred=model(img)
    # print(model)
    # pre=s_model(img)
    # print(pre.shape)
    # print(pred.shape)
    # print(s_model)
    for k in model.state_dict().keys():
        print(k)
    flops,params=0,0
    # with torch.no_grad():
    #     model.eval()

    #     #with torch.cuda.device(0):
    #     net = model
    #     flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True,
    #                                                 print_per_layer_stat=True)  # 不用写batch_size大小，默认batch_size=1
    #     print('Flops:  ' + flops)
    #     print('Params: ' + params)
    load_rm_block_state_dict(s_model, model.state_dict(), rm_blocks)


