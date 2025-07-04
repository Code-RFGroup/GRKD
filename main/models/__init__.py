from __future__ import absolute_import

import torch

from . import resnet
from . import mobilenetv2
from . import mobvit_change
from compute_flops import compute_MACs_params
from speed import eval_speed
from . import edgevit
from . import repvit
from . import mobile_vit
def build_teacher(model, num_classes, teacher='', cuda=True):
    if 'resnet50' in model:
        origin_model = resnet.resnet50(pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 256, 256))
        all_blocks = resnet.get_blocks_to_drop(origin_model)
    elif 'resnet34' in model:
        origin_model = resnet.resnet34(pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = resnet.get_blocks_to_drop(origin_model)
    elif 'mobilenet_v2' in model:
        origin_model = mobilenetv2.mobilenet_v2(pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 256, 256))
        all_blocks = mobilenetv2.get_blocks_to_drop()
    elif 'mobilelk' in model:
        origin_model = mobvit_change.mobilelk(pretrained=True)
        summary_data = torch.zeros((1, 3, 256, 256))
        all_blocks = mobvit_change.get_blocks_to_drop()
    elif 'mobilevit' in model:
        origin_model = mobile_vit.mobilevit(pretrained=True)
        summary_data = torch.zeros((1, 3, 256, 256))
        all_blocks = mobile_vit.get_blocks_to_drop()
    elif 'edge' in model:
        origin_model = edgevit.edgevit(pretrained=True, num_classes=num_classes)
        # if len(rm_blocks)==1:
        #     one_model=mobilenetv2.mobilenet_v2_one_blocks(rm_blocks, pretrained=True,num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 256, 256))
        all_blocks = edgevit.get_blocks_to_drop()
    elif 'repvit' in model:
        origin_model=repvit.repvit_m0_6(pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 256, 256))
        all_blocks = repvit.get_blocks_to_drop()
    else:
        raise ValueError(model)

    print('=> origin_model: {}'.format(all_blocks))
    origin_MACs, origin_Params = compute_MACs_params(origin_model, summary_data)
    MACs_str = f'MACs={origin_MACs:.3f}G'
    Params_str = f'Params={origin_Params:.3f}M'
    
    latency_data = torch.randn(1, 3, 256, 256)
    if cuda:
        origin_model.cuda()
        latency_data = latency_data.cuda()
    latency = eval_speed(origin_model, latency_data) * 1000
    latency_str = f'Lat={latency:.3f}ms'
    print(f'=> origin_model: {latency_str}, {MACs_str}, {Params_str}')

    return origin_model, all_blocks, latency

def build_student(model, rm_blocks, num_classes, state_dict_path='', teacher='', no_pretrained=False, cuda=True):
    if state_dict_path:
        print('=> load check_point from {}'.format(state_dict_path))
        check_point = torch.load(state_dict_path)
        rm_blocks = check_point['rm_blocks']
        state_dict = check_point['state_dict']

    if 'resnet50' in model:
        pruned_model = resnet.resnet50_rm_blocks(rm_blocks, pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 256, 256))
        if len(rm_blocks)==1:
            one_model=resnet.resnet50_one_blocks(rm_blocks, pretrained=True,num_classes=num_classes)
    elif 'resnet34' in model:
        pruned_model = resnet.resnet34_rm_blocks(rm_blocks, pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 224, 224))
    elif 'mobilenet_v2' in model:
        pruned_model = mobilenetv2.mobilenet_v2_rm_blocks(rm_blocks, pretrained=True, num_classes=num_classes)
        if len(rm_blocks)==1:
            one_model=mobilenetv2.mobilenet_v2_one_blocks(rm_blocks, pretrained=True,num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 256, 256))
    elif 'mobilelk' in model:
        pruned_model = mobvit_change.mobilk_v2_rm_blocks(rm_blocks, pretrained=True)
        if len(rm_blocks)==1:
            one_model=mobvit_change.mobilk_v2_one_blocks(rm_blocks, pretrained=True)
        summary_data = torch.zeros((1, 3, 256, 256))
    elif 'mobilevit' in model:
        pruned_model = mobile_vit.mobvit_rm_blocks(rm_blocks, pretrained=True, num_classes=num_classes)
        # if len(rm_blocks)==1:
        #     one_model=mobilenetv2.mobilenet_v2_one_blocks(rm_blocks, pretrained=True,num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 256, 256))
    elif 'edge' in model:
        pruned_model = edgevit.edge_rm_blocks(rm_blocks, pretrained=True, num_classes=num_classes)
        # if len(rm_blocks)==1:
        #     one_model=mobilenetv2.mobilenet_v2_one_blocks(rm_blocks, pretrained=True,num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 256, 256))
    elif 'repvit' in model:
        pruned_model=repvit.repvit_rm_blocks(rm_blocks, pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 256, 256))
    else:
        raise ValueError(model)

    if state_dict_path and not no_pretrained:
        print('=> use pretrained weight from {}'.format(state_dict_path))
        pruned_model.load_state_dict(state_dict)
    else:
        print('=> use pretrained weight from teacher')

    print('=> remove blocks: {}'.format(rm_blocks))

    pruned_MACs, pruned_Params = compute_MACs_params(pruned_model, summary_data)
    MACs_str = f'MACs={pruned_MACs:.3f}G'
    Params_str = f'Params={pruned_Params:.3f}M'

    latency_data = torch.randn(1, 3, 256, 256)
    if cuda:
        pruned_model.cuda()
        # if len(rm_blocks)==1:
        #     one_model.cuda()
        latency_data = latency_data.cuda()
    latency = eval_speed(pruned_model, latency_data) * 1000
    latency_str = f'Lat={latency:.3f}ms'
    print(f'=> pruned_model: {latency_str}, {MACs_str}, {Params_str}')
    # if len(rm_blocks)==1:
    #     return pruned_model, None, latency,one_model
    # else:
    return pruned_model, None, latency,None


def build_models(model, rm_blocks, num_classes=1000, state_dict_path='', teacher='', no_pretrained=False):
    if state_dict_path:
        print('=> load check_point from {}'.format(state_dict_path))
        check_point = torch.load(state_dict_path)
        rm_blocks = check_point['rm_blocks']
        state_dict = check_point['state_dict']

    if 'resnet50' in model:
        origin_model = resnet.resnet50(pretrained=True, num_classes=num_classes)
        pruned_model = resnet.resnet50_rm_blocks(rm_blocks, pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 256, 256))
        all_blocks = resnet.get_blocks_to_drop(origin_model)
    elif 'resnet34' in model:
        origin_model = resnet.resnet34(pretrained=True, num_classes=num_classes)
        pruned_model = resnet.resnet34_rm_blocks(rm_blocks, pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 224, 224))
        all_blocks = resnet.get_blocks_to_drop(origin_model)
    elif 'mobilenet_v2' in model:
        origin_model = mobilenetv2.mobilenet_v2(pretrained=True, num_classes=num_classes)
        pruned_model = mobilenetv2.mobilenet_v2_rm_blocks(rm_blocks, pretrained=True, num_classes=num_classes)
        summary_data = torch.zeros((1, 3, 256, 256))
        all_blocks = mobilenetv2.get_blocks_to_drop()
    else:
        raise ValueError(model)

    if state_dict_path and not no_pretrained:
        print('=> use pretrained weight from {}'.format(state_dict_path))
        pruned_model.load_state_dict(state_dict)
    else:
        print('=> use pretrained weight from teacher')

    print('=> origin_model: {}'.format(all_blocks))
    print('=> remove blocks: {}'.format(rm_blocks))

    origin_MACs, origin_Params = compute_MACs_params(origin_model, summary_data)
    pruned_MACs, pruned_Params = compute_MACs_params(pruned_model, summary_data)
    reduce_MACs = (origin_MACs-pruned_MACs) / origin_MACs * 100
    reduce_Params = (origin_Params - pruned_Params) / origin_Params * 100
    MACs_str = f'MACs={pruned_MACs:.3f}G (prune {reduce_MACs:.2f}%)'
    Params_str = f'Params={pruned_Params:.3f}M (prune {reduce_Params:.2f}%)'
    print(f'=> pruned_model: {MACs_str}, {Params_str}')

    return pruned_model, origin_model, rm_blocks
