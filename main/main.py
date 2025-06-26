import imp
import os
import argparse
import logging
import  seaborn
import collections
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets, transforms
import random
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import time
import torch.optim as optim
import torch.nn.functional as F

from models import *
import dataset
from practise import Practise_one_block, Practise_all_blocks
from finetune import end_to_end_finetune, validate
def set_seed(seed=42): # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #根据文档，torch.manual_seed(seed)应该已经为所有设备设置seed
    #但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #cuDNN在使用deterministic模式时（下面两行），可能会造成性能下降（取决于model）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# Prune settings
#set_seed(4093)
# Prune settings
parser = argparse.ArgumentParser(description='Accelerate networks by PRACTISE')
parser.add_argument('--dataset', type=str, default='ip22',
                    help='training dataset (default: imagenet_fewshot)')
parser.add_argument('--eval-dataset', type=str, default='ip22',
                    help='training dataset (default: imagenet)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num_sample', type=int, default=500,
                    help='number of samples for training')
parser.add_argument('--model', default='mobilelk', type=str, #mobilelk #mobilenet_v2
                    help='model name (default: resnet34)')
parser.add_argument('--teacher', default='', type=str, metavar='PATH',
                    help='path to the pretrained teacher model (default: none)')
parser.add_argument('--save', default='results', type=str, metavar='PATH',
                    help='path to save pruned model (default: results)')
parser.add_argument('--state_dict_path', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--no-pretrained', action='store_true', default=False,
                    help='do not use pretrained weight')

parser.add_argument('--rm_blocks', default='features.10', type=str,
                    help='names of removed blocks, split by comma')
parser.add_argument('--practise', default='', type=str,
                    help='blocks for practise', choices=['', 'one', 'all'])
parser.add_argument('--FT', default='BP', type=str,
                    help='method for finetuning', choices=['', 'BP', 'MiR'])
parser.add_argument('--lambda_kd', type=int, default=1.0,
                    help='')
parser.add_argument('--T', type=int, default=4.0,
                    help='temperature')
parser.add_argument('--opt', default='SGD', type=str,
                    help='opt method (default: SGD)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.02)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--batch-size', type=int, default=32,
                    help='number of batch size')
parser.add_argument('--epoch', type=int, default=10,
                    help='number of epoch')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--seed', type=int, default=66,
                    help='seed')

transform_train = transforms.Compose([
     transforms.Resize([256,256]),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.RandomRotation(90),
     transforms.ToTensor(),
#transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))

     #transforms.Normalize((0.3028, 0.3741, 0.3199), (0.1183, 0.1072, 0.0937))
 ])

transform_test = transforms.Compose([
     transforms.Resize([256,256]),
     transforms.ToTensor(),

     #transforms.Normalize((0.3028, 0.3741, 0.3199), (0.1183, 0.1072, 0.0937))
 ])
trainset =ImageFolder(root=r'../cifar-100-python/train',transform=transform_train)
#trainset =ImageFolder(root=r'../ip22v1/train',transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
valset =ImageFolder(root=r'../cifar-100-python/val',transform=transform_test)
testset =ImageFolder(root=r'../cifar-100-python/test',transform=transform_test) 
val_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True,num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True,num_workers=2)
metric_loader=val_loader
#metric_loader=test_loader
# visualize the difference between the teacher's output logits and the student's
def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / stdv

def get_output_metric_oneindex(model, val_loader, num_classes=100, ifstand=False):
    model.eval()
    index = 7#7, 12
    all_preds, all_labels = None, None
    sample = None
    label = None
    with torch.no_grad():
        for i, (data, labels) in tqdm(enumerate(val_loader)):
            if i != index:
                continue
            sample = data.cuda()
            data=data.cuda()
            labels=labels.cuda()
            label = labels.cuda()
#             plt.imshow(sample.permute(1,2,0))
            outputs = model(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if ifstand:
                preds = normalize(outputs)
            else:
                preds = outputs
            all_preds = preds.data.cpu().numpy()
            all_labels = labels.data.cpu().numpy()
    print(all_preds.shape)
    print(label)
    print(all_preds.mean(-1).max())
    print(all_preds.std(-1).max())
    return all_preds[0], all_preds.std(-1).max()
def get_tea_stu_diff_oneindex(tea, stu, val_loader ,ifstand=False):
    #cfg.defrost()
    #cfg.freeze()
    #train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    model = stu
    tea_model = tea
    print("load model successfully!")
    ms, maxstd_s = get_output_metric_oneindex(model, val_loader, ifstand=ifstand)
    mt, maxstd_t = get_output_metric_oneindex(tea_model, val_loader, ifstand=ifstand)
    font_size=18
    x = list(range(100)) #["1"]*100
    xlabels = []
    for ii in range(10):
        xlabels += [ii*10]
        xlabels += [''] * 9
    plt.figure(figsize=(9,5))
    ax = seaborn.barplot(x=np.array(x), y=mt, color='blue' )
    ax = seaborn.barplot(x=np.array(x), y=ms, color='red')
    ax.tick_params(bottom=False,top=False,left=False,right=False)
    ax.set_xticklabels(xlabels, fontsize=17)
    topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
    bottombar = plt.Rectangle((0,0),1,1,fc='blue',  edgecolor = 'none')
    l = plt.legend([bottombar, topbar], ['teacher, std.=%.2f'%(maxstd_t), 'student, std.=%.2f'%(maxstd_s)], loc=1, ncol = 1, prop={'size':font_size})
    l.draw_frame(False)
    plt.yticks(fontsize=17)
    plt.ylim(-5,12.5)
    plt.tight_layout()
    plt.xlabel("class category", fontsize = font_size)
    plt.ylabel("logit value", fontsize = font_size)
    plt.show()
    plt.close()
    
#     plt.figure(figsize=(7,5))
#     ax.set_xticklabels(xlabels, fontsize=17)
#     plt.yticks(fontsize=17)
#     plt.xlabel("class", fontsize = 20)
#     plt.ylabel("logit", fontsize = 20)
#     plt.show()
    print(ms.mean())
    print(mt.mean())
    return ms, mt
def main():
    global args
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    
    args.save = os.path.join(args.save, r"{}_{}_{}_{}/{}_{}".format(
        args.model, args.dataset, args.practise, args.FT, args.num_sample, args.seed))
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    LOG = logging.getLogger('main')
    time_now = datetime.now()
    logfile = os.path.join(args.save, 'log_{date:%Y-%m-%d_%H:%M:%S}.txt'.format(
        date=time_now))
    
    # logfile=logfile.replace(r'\\', "/")
    # print(logfile)
    #print(logging)
    FileHandler = logging.FileHandler(logfile, mode='w')
    LOG.addHandler(FileHandler)

    #print('22222')
    import builtins as __builtin__
    #print('333')
    builtin_print = __builtin__.print

    #print('333')
    __builtin__.print = LOG.info
    #print('333')
    args.eval_freq=500
    
    args.num_classes=100
    print(args)
    origin_model, all_blocks, origin_lat = build_teacher(
        args.model, args.num_classes, teacher=args.teacher, cuda=args.cuda
    )
    validate(test_loader, origin_model)
    print('--------------')
    if args.rm_blocks:
        rm_blocks = args.rm_blocks.split(',')
    else:
        rm_blocks = []
    if args.practise == 'one':
        assert len(rm_blocks) == 1
        pruned_model, _ = Practise_one_block(rm_blocks[0], origin_model, origin_lat, train_loader, metric_loader, args)
    elif args.practise == 'all':
        #pruned_model, rm_blocks = Practise_all_blocks(all_blocks, origin_model, origin_lat, train_loader, metric_loader, args)
        pruned_model, rm_blocks, delete_model = Practise_all_blocks(all_blocks, origin_model, origin_lat, train_loader,
                                                                   metric_loader, args)
    else:
        pruned_model, _, pruned_lat,one_model = build_student(
            args.model, rm_blocks, args.num_classes, 
            state_dict_path=args.state_dict_path, teacher=args.teacher, cuda=args.cuda
        )
        lat_reduction = (origin_lat - pruned_lat) / origin_lat * 100
        print(f'=> latency reduction: {lat_reduction:.2f}%')
    time_now=datetime.now()
    if args.FT:
        validate(test_loader, pruned_model)
        print("=> finetune:")
        #end_to_end_finetune(train_loader, val_loader, delete_model, origin_model,pruned_model,args,False )
        end_to_end_finetune(train_loader, val_loader,  pruned_model, origin_model,delete_model,args,True )
        #end_to_end_finetune(train_loader, val_loader, pruned_model, origin_model, args)
        validate(test_loader, pruned_model)
        #validate(test_loader,origin_model)
        get_tea_stu_diff_oneindex(origin_model, pruned_model, val_loader, ifstand=False)
        save_path = 'check_point_{:%Y-%m-%d_%H:%M:%S}.tar'.format(time_now)
        save_path = os.path.join(args.save, save_path)
        check_point = {
            'state_dict': pruned_model.state_dict(),
            'rm_blocks': rm_blocks,
        }
        checkpoint={'state_dict':delete_model.state_dict(),

'rm_blocks':list(set(all_blocks)-set(rm_blocks))
                    }
        torch.save(checkpoint,os.path.join(args.save,'_c'))
        torch.save(check_point, save_path)



if __name__ == '__main__':
    main()
