import os
import gc
import argparse
import logging
import collections
from datetime import datetime
import time
import torch
from torch import nn
import numpy as np
import torch
from finetune import accuracy
from models import *
import dataset
from finetune import AverageMeter, validate, accuracy
from compute_flops import compute_MACs_params
from models.AdaptorWarp import AdaptorWarp
import torch.nn.functional as F
def relation_similarity_metric(teacher, student, batch_data,model_name):
    image, label = batch_data
    image,label=image.cuda(),label.cuda()
    #image=batch_data
    #print(model_name)
    # Forward pass
    t_feats = teacher.forward_features(image)
    s_feats = student.forward_features(image)
    #print(t_feats.shape)
    # Get activation before average pooling
    #t_feat = t_feats[1]
   # s_feat = s_feats[1]
    # Compute batch similarity
    return  batch_similarity(t_feats, s_feats)

def batch_similarity(f_t, f_s):
    bsz=f_s.shape[0]
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
    #print(G_t)
    #print(G_t.shape)
    #print(G_diff)
    #print((G_diff * G_diff))
    return (G_diff * G_diff).view(-1, 1).sum() / (bsz * bsz)
 
def semantic_similarity_metric(teacher, student, batch_data,model_name):
    criterion = nn.CrossEntropyLoss() 
    image, label = batch_data 
    image,label=image.cuda(),label.cuda()
    # Forward once.
    t_logits,t_features = teacher.forward(image)
    s_logits,s_features = student.forward(image)
    #print(t_logits,label)
    # Backward once.
    #print(t_logits)
    criterion(t_logits, label).backward()
    criterion(s_logits, label).backward()
    #print(model_name)
    # Grad-cam of fc layer.
    if 'repvit' in model_name:
        t_grad_cam = teacher.classifier.classifier.l.weight.grad
        s_grad_cam = student.classifier.classifier.l.weight.grad
    elif 'mobilenet_v2' in model_name:
        #print(model_name)
        t_grad_cam = teacher.classifier[1].weight.grad
        s_grad_cam = student.classifier[1].weight.grad
    elif 'edge' in model_name:
        t_grad_cam = teacher.head.weight.grad
        s_grad_cam = student.head.weight.grad
    elif 'resnet50' or 'resnet34' in model_name:
        #print(model_name+'!!!!!!!!!!!!!')
        t_grad_cam = teacher.fc.weight.grad
        s_grad_cam = student.fc.weight.grad





    elif 'mobilevit' in model_name:
        t_grad_cam = teacher.to_logits[2].weight.grad
        s_grad_cam = student.to_logits[2].weight.grad
    print(t_grad_cam.shape)
    print('11111')
    # # Compute channel-wise similarity
    return  channel_similarity(t_grad_cam, s_grad_cam)

def channel_similarity(f_t, f_s):
    bsz, ch = f_s.shape[0], f_s.shape[1]
    # Reshape
    print(f_t.shape)#N C N is classes
    print('222')
    f_s = f_s.view(bsz, ch, -1)
    f_t = f_t.view(bsz, ch, -1)
    print(f_t.shape)#N C 1
    print('333')
    # Get channel-wise similarity matrix
    emd_s = torch.bmm(f_s, f_s.permute(0, 2, 1))
    emd_s = F.normalize(emd_s, dim=2)
    emd_t = torch.bmm(f_t, f_t.permute(0, 2, 1))
    emd_t = F.normalize(emd_t, dim=2)
    # Produce L_2 distance
    G_diff = emd_s - emd_t
    print(emd_t.shape)#N C C
    print('444')
    # print(G_diff* G_diff)
    print((G_diff*G_diff).shape)#N C C
    #print((G_diff * G_diff).view(bsz, -1).sum())
    return (G_diff * G_diff).view(bsz, -1).sum() / (ch * bsz*bsz)
def Practise_one_block(rm_block, origin_model, origin_lat, train_loader, metric_loader, args):
    gc.collect()
    torch.cuda.empty_cache()

    pruned_model, _, pruned_lat,one_model = build_student(
        args.model, [rm_block], args.num_classes, 
        state_dict_path=args.state_dict_path, teacher=args.teacher, cuda=args.cuda
    )
    lat_reduction = (origin_lat - pruned_lat) / origin_lat * 100
    print(f'=> latency reduction: {lat_reduction:.2f}%')

    print("ACC w/o Recovering:")
    #re=relation_similarity_metric(origin_model,pruned_model,metric_loader)
    #se=semantic_similarity_metric(origin_model,pruned_model,metric_loader)
    #a_acc=metric(metric_loader, origin_model, origin_model)
    #c_acc=metric(metric_loader, pruned_model, origin_model)
    #b_acc=metric(metric_loader, one_model, origin_model)

    #pruned_model_adaptor = AdaptorWarp(pruned_model)

    #print('adaptore',prun)
    start_time = time.time()
    #Practise_recover(train_loader, origin_model, pruned_model, [rm_block], args)
    #Practise_recover(train_loader, origin_model, pruned_model_adaptor,one_model, [rm_block], args)
    print("Total time: {:.3f}s".format(time.time() - start_time))
    recoverability=metric_two(metric_loader, pruned_model, origin_model,args.model)
    print("Metric w/ Recovering:")
    #recoverability=b_acc+a_acc-c_acc
    #recoverability = metric(metric_loader, one_model, origin_model)-metric(metric_loader, pruned_model_adaptor, origin_model)


    score = recoverability / 1.0 #lat_reduction  这里去掉了latency!!!!!
    print(f"{rm_block} -> {recoverability:.4f}/{lat_reduction:.2f}={score:.5f}")
    return pruned_model, (recoverability, lat_reduction, score)

def Practise_all_blocks(rm_blocks, origin_model, origin_lat, train_loader, metric_loader, args):
    recoverabilities = dict()
    for rm_block in rm_blocks:
        _, results = Practise_one_block(rm_block, origin_model, origin_lat, train_loader, metric_loader, args)
        recoverabilities[rm_block] = results

    print('-' * 50)
    sort_list = []
    for block in recoverabilities:
        recoverability, lat_reduction, score = recoverabilities[block]
        print(f"{block} -> {recoverability:.4f}/{lat_reduction:.2f}={score:.5f}")
        sort_list.append([score, block])
    print('-' * 50)
    print('=> sorted')
    sort_list.sort()
    for score, block in sort_list:
        print(f"{block} -> {score:.4f}")
    print('-' * 50)
    print(f'=> scores of {args.model} (#data:{args.num_sample}, seed={args.seed})')
    print('Please use this seed to recover the model!')
    print('-' * 50)

    drop_blocks = []
    if args.rm_blocks.isdigit():
        for i in range(int(args.rm_blocks)):
            drop_blocks.append(sort_list[i][1])
    pruned_model, _, pruned_lat,one_model = build_student(
        args.model, drop_blocks, args.num_classes, 
        state_dict_path=args.state_dict_path, teacher=args.teacher, cuda=args.cuda
    )
    delete_model, _, pruned_lat,one_model = build_student(
        args.model, list(set(rm_blocks)-set(drop_blocks)), args.num_classes,
        state_dict_path=args.state_dict_path, teacher=args.teacher, cuda=args.cuda
    )
    lat_reduction = (origin_lat - pruned_lat) / origin_lat * 100
    print(f'=> latency reduction: {lat_reduction:.2f}%')
    return pruned_model, drop_blocks,delete_model
def insert_one_block_adaptors_for_mobilelk(origin_model, prune_model, rm_block, params, args):#features.0 和features10有问题
    origin_named_modules = dict(origin_model.named_modules())
    pruned_named_modules = dict(prune_model.model.named_modules())
    print('-' * 50)
    print('=> {}'.format(rm_block))
    has_rm_count = 0
    rm_channel = origin_named_modules[rm_block].out_channels
    key_items = rm_block.split('.')
    block_id = int(key_items[1])

    pre_block_id = block_id-has_rm_count-1
    while pre_block_id >= 0:
        pruned_module = pruned_named_modules[f'features.{pre_block_id}']
        if pre_block_id==0:
            pass
        else:
            if rm_channel != pruned_module.out_channels:
                break
        if rm_block=='features.1':
            last_conv_key = 'features.{}.0'.format(pre_block_id)
        else:
            last_conv_key = 'features.{}.conv.6'.format(pre_block_id)#在卷积后面加
        conv = prune_model.add_afterconv_for_conv(last_conv_key)
        params.append({'params': conv.parameters()})
        pre_block_id -= 1
        # break

    after_block_id = block_id - has_rm_count
    while after_block_id < 11:
        if rm_block=='features.10':
            pruned_module=pruned_named_modules[f'to_logits.0.0']
        else:
            pruned_module = pruned_named_modules[f'features.{after_block_id}']
        if rm_block=='features.10':
            after_conv_key = 'to_logits.0.0'#在卷积前面加
        else:
            after_conv_key = 'features.{}.conv.0'.format(after_block_id)#在卷积前面加
        conv = prune_model.add_preconv_for_conv(after_conv_key)
        params.append({'params': conv.parameters()})
        if rm_block=='features.10':
            break
        else:
            if rm_channel != pruned_module.out_channels:
                break
        after_block_id += 1
        # break

    has_rm_count += 1

def insert_one_block_adaptors_for_mobilenet(origin_model, prune_model, rm_block, params, args):
    origin_named_modules = dict(origin_model.named_modules())
    pruned_named_modules = dict(prune_model.model.named_modules())

    print('-' * 50)
    print('=> {}'.format(rm_block))#
    has_rm_count = 0
    rm_channel = origin_named_modules[rm_block].out_channels
    key_items = rm_block.split('.')
    block_id = int(key_items[1])

    pre_block_id = block_id-has_rm_count-1
    while pre_block_id > 0:
        pruned_module = pruned_named_modules[f'features.{pre_block_id}']
        if rm_channel != pruned_module.out_channels:
            break
        last_conv_key = 'features.{}.conv.2'.format(pre_block_id)
        conv = prune_model.add_afterconv_for_conv(last_conv_key)
        params.append({'params': conv.parameters()})
        pre_block_id -= 1
        # break

    after_block_id = block_id - has_rm_count
    while after_block_id < 18:
        pruned_module = pruned_named_modules[f'features.{after_block_id}']
        after_conv_key = 'features.{}.conv.0.0'.format(after_block_id)
        conv = prune_model.add_preconv_for_conv(after_conv_key)
        params.append({'params': conv.parameters()})
        if rm_channel != pruned_module.out_channels:
            break
        after_block_id += 1
        # break

    has_rm_count += 1


   

def insert_one_block_adaptors_for_resnet(prune_model, rm_block, params, args):
    pruned_named_modules = dict(prune_model.model.named_modules())
    if 'layer1.0.conv2' in pruned_named_modules:
        last_conv_in_block = 'conv2'
    elif 'layer1.0.conv3' in pruned_named_modules:
        last_conv_in_block = 'conv3'
    else:
        raise ValueError("This is not a ResNet.")

    print('-' * 50)
    print('=> {}'.format(rm_block))
    layer, block = rm_block.split('.')
    rm_block_id = int(block)
    assert rm_block_id >= 1

    downsample = '{}.0.downsample.0'.format(layer)
    if downsample in pruned_named_modules:
        conv = prune_model.add_afterconv_for_conv(downsample)
        if conv is not None:
            params.append({'params': conv.parameters()})

    for origin_block_num in range(rm_block_id):
        last_conv_key = '{}.{}.{}'.format(layer, origin_block_num, last_conv_in_block)
        conv = prune_model.add_afterconv_for_conv(last_conv_key)
        if conv is not None:
            params.append({'params': conv.parameters()})

    for origin_block_num in range(rm_block_id+1, 100):
        pruned_output_key = '{}.{}.conv1'.format(layer, origin_block_num-1)
        if pruned_output_key not in pruned_named_modules:
            break
        conv = prune_model.add_preconv_for_conv(pruned_output_key)
        if conv is not None:
            params.append({'params': conv.parameters()})

    # next stage's conv1
    next_layer_conv1 = 'layer{}.0.conv1'.format(int(layer[-1]) + 1)
    if next_layer_conv1 in pruned_named_modules:
        conv = prune_model.add_preconv_for_conv(next_layer_conv1)
        if conv is not None:
            params.append({'params': conv.parameters()})

    # next stage's downsample
    next_layer_downsample = 'layer{}.0.downsample.0'.format(int(layer[-1]) + 1)
    if next_layer_downsample in pruned_named_modules:
        conv = prune_model.add_preconv_for_conv(next_layer_downsample)
        if conv is not None:
            params.append({'params': conv.parameters()})


def insert_all_adaptors_for_resnet(origin_model, prune_model, rm_blocks, params, args):
    rm_blocks_for_prune = []
    rm_blocks.sort()
    rm_count = [0, 0, 0, 0]
    for block in rm_blocks:
        layer, i = block.split('.')
        l_id = int(layer[-1])
        b_id = int(i)
        prune_b_id = b_id - rm_count[l_id-1]
        rm_count[l_id-1] += 1
        rm_block_prune = f'{layer}.{prune_b_id}'
        rm_blocks_for_prune.append(rm_block_prune)
    for rm_block in rm_blocks_for_prune:
        insert_one_block_adaptors_for_resnet(prune_model, rm_block, params, args)


def Practise_recover(train_loader, origin_model, prune_model,one_model, rm_blocks, args):
    params = []

    if 'mobilenet' in args.model:
        assert len(rm_blocks) == 1
        insert_one_block_adaptors_for_mobilenet(origin_model, prune_model, rm_blocks[0], params, args)
    elif 'mobilelk' in args.model:
        insert_one_block_adaptors_for_mobilelk(origin_model, prune_model, rm_blocks[0], params, args)
        #pass
    else:
        insert_all_adaptors_for_resnet(origin_model, prune_model, rm_blocks, params, args)
    #print(prune_model)
        #     self.module2preconvs = dict()
        # self.name2preconvs = dict()
        # self.prehandles = dict()
        # self.module2afterconvs = dict()
        # self.name2afterconvs = dict()
        # self.afterhandles = dict()
    # print(prune_model.module2preconvs)#{Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)}
    # print(prune_model.name2preconvs)#{'features.3.conv.0.0': Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)}
    # print(prune_model.prehandles)#{Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False): <torch.utils.hooks.RemovableHandle object at 0x0000013984D63E88>}
    #print(params)
    if args.opt == 'SGD':
        optimizer_one = torch.optim.SGD(one_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        #optimizer = torch.optim.SGD(prune_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("{} not found".format(args.opt))

    recover_time = time.time()
    train(train_loader, optimizer_one, one_model, origin_model, args)
    train(train_loader, optimizer, prune_model, origin_model, args)
    print("compute recoverability {} takes {}s".format(rm_blocks, time.time() - recover_time))


def train(train_loader, optimizer, model, origin_model, args):
    # Data loading code
    end = time.time()
    #criterion = torch.nn.MSELoss(reduction='mean')
    criterion = torch.nn.CrossEntropyLoss()
    # switch to train mode
    origin_model.cuda()
    origin_model.eval()
    model.cuda()
    model.eval()
    #model.get_feat = 'pre_GAP'
    model.get_feat='after_GAP'
    #origin_model.get_feat = 'pre_GAP'
    origin_model.get_feat='after_GAP'

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(0.4 * args.epoch), gamma=0.1)

    torch.cuda.empty_cache()
    iter_nums = 0
    finish = False
    while not finish:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        for batch_idx, (data, target) in enumerate(train_loader):
            iter_nums += 1
            if iter_nums > args.epoch:
                finish = True
                break
            # measure data loading time
            data_time.update(time.time() - end)
            data = data.cuda()
            target=target.cuda()
            with torch.no_grad():
                t_output, t_features = origin_model(data)
            optimizer.zero_grad()
            output, s_features = model(data)
            loss=criterion(output,target)
            #loss = criterion(s_features, t_features)
            losses.update(loss.data.item(), data.size(0))
            loss.backward()
            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if iter_nums % 50 == 0:
                print('Train: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(
                   iter_nums, args.epoch, batch_time=batch_time,
                   data_time=data_time, losses=losses))
            scheduler.step()

def metric_two(metric_loader, model, origin_model,model_name):
    #criterion = torch.nn.MSELoss(reduction='mean')
    criterion = torch.nn.CrossEntropyLoss()

    # switch to train mode
    origin_model.cuda()
    origin_model.eval()
    #origin_model.get_feat = 'pre_GAP'
    origin_model.get_feat ='after_GAP'
    model.cuda()
    model.eval()
    #model.get_feat = 'pre_GAP'
    model.get_feat ='after_GAP'
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    #re=relation_similarity_metric(origin_model,pruned_model,metric_loader)
    #se=semantic_similarity_metric(origin_model,pruned_model,metric_loader)

    end = time.time()
    #print('len metric_loader',len(metric_loader))
    re_all=0.0
    se_all=0.0
    for i, batch_data in enumerate(metric_loader):

        re=relation_similarity_metric(origin_model,model,batch_data,model_name)
        se=semantic_similarity_metric(origin_model,model,batch_data,model_name)
        #data = data.cuda()
        #target=target.cuda()
        re_all=re_all+re.item()
        se_all=se_all+se.item()
        data_time.update(time.time() - end)

        #losses.update(loss.data.item(), data.size(0))
        # measure elapsed time
        #_, predicted = torch.max(s_output.data, 1)


        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            print('Metric: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                're {re:.4f} \t'
                'se {se:.3f} \t'.format(
                i, len(metric_loader), batch_time=batch_time,
                data_time=data_time, re=re,se=se))
    print('re_all:{}'.format(re_all/len(metric_loader)))
    print('se_all:{}'.format(se_all/len(metric_loader)))
    #return losses.avg
    #return re_all/len(metric_loader)#+se_all/len(metric_loader)
    return se_all/len(metric_loader)
def metric(metric_loader, model, origin_model):
    #criterion = torch.nn.MSELoss(reduction='mean')
    criterion = torch.nn.CrossEntropyLoss()

    # switch to train mode
    origin_model.cuda()
    origin_model.eval()
    #origin_model.get_feat = 'pre_GAP'
    origin_model.get_feat ='after_GAP'
    model.cuda()
    model.eval()
    #model.get_feat = 'pre_GAP'
    model.get_feat ='after_GAP'
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    #print('len metric_loader',len(metric_loader))
    for i, (data, target) in enumerate(metric_loader):
        with torch.no_grad():
            data = data.cuda()
            target=target.cuda()
            data_time.update(time.time() - end)
            t_output, t_features = origin_model(data)
            s_output, s_features = model(data)
            loss=criterion(s_output,target)
            #loss = criterion(s_features, t_features)

        losses.update(loss.data.item(), data.size(0))
        # measure elapsed time
        #_, predicted = torch.max(s_output.data, 1)
        prec1, prec5 = accuracy(s_output.data, target, topk=(1, 5))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))
        #print(predicted)
        #print(target)
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            print('Metric: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(metric_loader), batch_time=batch_time,
                data_time=data_time, losses=losses,top1=top1, top5=top5))

    print(' * Metric Loss {loss.avg:.4f}'.format(loss=losses))
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    #return losses.avg
    return top1.avg




if __name__ == '__main__':
    main()
