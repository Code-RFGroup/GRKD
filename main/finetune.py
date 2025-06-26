import os
import logging
import collections
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
import torch.optim as optim
from collections import deque
def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)
def std_loss(logits_students, logits_teachers,temperature):
    logits_student = normalize(logits_students)
    logits_teacher = normalize( logits_teachers)
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


kl_max = 0.0
a_kl = 0.0
b_kl = 0.0
class HybridMaxTracker:
    def __init__(self, short_win=100, long_win=500, ema_gamma=0.95):
        self.short_window = deque(maxlen=short_win) # 捕捉近期波动
        self.long_window = deque(maxlen=long_win) # 保留历史极值
        self.ema_max = 0.0
        self.gamma = ema_gamma
    def update(self, new_kl): # 更新窗口
        self.short_window.append(new_kl)
        if new_kl > 0.7 * max(self.long_window, default=new_kl):
            self.long_window.append(new_kl) # 动态计算当前极值
        current_short_max = max(self.short_window)
        self.ema_max = self.gamma*self.ema_max + (1-self.gamma)*current_short_max
        return max(current_short_max, self.ema_max, max(self.long_window))
kl_tracker = HybridMaxTracker(short_win=100, long_win=200, ema_gamma=0.95)

delta=0.0
sigma=0.0
def dkd_loss(logits_student, logits_teacher, target, alpha, temperature, base_k,iter_nums,flag):
    batch_size, num_classes = logits_student.shape
    logits_student = normalize(logits_student)
    logits_teacher = normalize( logits_teacher)

    # 获取目标类掩码（保持原逻辑）
    gt_mask = _get_gt_mask(logits_student, target)  # [B, C]
    other_mask = _get_other_mask(logits_student, target)  # [B, C]
    global  kl_max
    if flag:
        kl_max=0.0
    ######################################
    # 动态能力差距计算（新增模块）
    ######################################
    def compute_capacity_gap():
        global kl_max
        global  a_kl
        global  b_kl
        global kl_tracker

        with torch.no_grad():
            # 使用KL散度衡量模型差距
            #prob_s_1=normalize(logits_student)
            #prob_t_1=normalize(logits_teacher)
            prob_s = F.softmax(logits_student, dim=1)
            prob_t = F.softmax(logits_teacher, dim=1)
            kl_div = F.kl_div(prob_s.log(), prob_t, reduction='batchmean')
            if torch.isnan(kl_div) or torch.isinf(kl_div):
                kl_div = kl_max

            kl_max = max(kl_div, kl_max)
           # kl_max = kl_tracker.update(kl_div)
            #print(kl_div, kl_max)
            a_kl=kl_div
            b_kl=kl_max
            # return torch.sigmoid(kl_div)
            return kl_div / (kl_max + 1e-8)  # 归一化到[0,1]
    if iter_nums<=1000:
        dynamic_k=base_k
        gap_score=1.0
    else:
        gap_score = compute_capacity_gap()
        dynamic_k = int(base_k + (1 - gap_score) * (num_classes - base_k - 1))
        '''
        ######################################
        # TCKD部分（保持原样）
        ######################################
    tckd_loss = _calc_tckd_loss(logits_student, logits_teacher, gt_mask, other_mask, temperature)

    ######################################
    # 改进的NCKD部分（自适应Top-K）
    ######################################
    teacher_logits_nontarget = logits_teacher.masked_fill(gt_mask.bool(), -float('inf'))

    # 使用Gumbel-Softmax生成可微Top-K掩码（关键改进点）
    def gumbel_topk_mask(logits, k, temperature=0.5):
        gumbel = -torch.log(-torch.log(torch.rand_like(logits)))
        noisy_logits = (logits + gumbel) / temperature
        topk_probs = F.softmax(noisy_logits, dim=1)

        # 保留Top-K位置的梯度
        _, topk_indices = torch.topk(noisy_logits, k=k, dim=1)
        hard_mask = torch.zeros_like(logits).scatter(1, topk_indices, 1.0)

        # 直通估计器技巧（Straight-Through Estimator）
        return (hard_mask - topk_probs).detach() + topk_probs

    # 生成自适应掩码
    topk_mask = gumbel_topk_mask(teacher_logits_nontarget, k=dynamic_k)
    nckd_mask = other_mask * topk_mask  # 组合非目标类掩码

    # 改进的NCKD计算（支持软掩码）
    nckd_loss = _calc_adaptive_nckd_loss(
        logits_student,
        logits_teacher,
        nckd_mask,
        temperature
    )
    '''
    ######################################
    # 统一蒸馏损失（合并TCKD和NCKD）
    ######################################
    def gumbel_topk_mask(logits, k, temp=0.5):
        gumbel = -torch.log(-torch.log(torch.rand_like(logits)))
        noisy_logits = (logits + gumbel) / temp
        topk_probs = F.softmax(noisy_logits, dim=1)
        _, topk_idx = torch.topk(noisy_logits, k=k, dim=1)
        hard_mask = torch.zeros_like(logits).scatter(1, topk_idx, 1.0)
        return (hard_mask - topk_probs).detach() + topk_probs

    # 生成联合蒸馏掩码
    mask = gumbel_topk_mask(logits_teacher, k=dynamic_k)
    ######################################
    # 统一KL损失计算
    ######################################
    with torch.no_grad():
        prob_t = F.softmax(logits_teacher / temperature, dim=1)

    prob_s = F.log_softmax(logits_student / temperature, dim=1)

    # 应用动态掩码（同时包含目标类和非目标类）
    loss = F.kl_div(prob_s * mask,
                    prob_t * mask,
                    reduction='batchmean') * (temperature ** 2)

    f6 = open("dymaic_k.txt", "a+")

    f6.write("current_k=%d,gap_score= %.6f,kl_div=%.5f,kl_max=%.5f" % (dynamic_k, gap_score,a_kl,b_kl))
    f6.write('\n')

    f6.close()
    return alpha * loss#alpha * tckd_loss #+8 * nckd_loss#alpha * loss#alpha * tckd_loss +8 * nckd_loss#


def _calc_tckd_loss(logits_s, logits_t, gt_mask, other_mask, temperature):
    """目标类知识蒸馏损失（保持原DKD实现）"""
    pred_t = F.softmax(logits_t / temperature, dim=1)
    pred_s = F.log_softmax(logits_s / temperature, dim=1)

    tckd_loss = F.kl_div(
        (pred_s * gt_mask).sum(1),  # 目标类概率求和
        (pred_t * gt_mask).sum(1),  # 对应目标类概率求和
        reduction='batchmean'
    ) * (temperature ** 2)

    return tckd_loss


def _calc_adaptive_nckd_loss(logits_s, logits_t, nckd_mask, temperature):
    """支持软掩码的NCKD计算"""
    # 分离教师梯度（保持原论文设定）
    with torch.no_grad():
        prob_t = F.softmax(logits_t / temperature, dim=1)

    prob_s = F.log_softmax(logits_s / temperature, dim=1)

    # 应用软掩码加权
    weighted_prob_t = prob_t * nckd_mask
    weighted_prob_s = prob_s * nckd_mask

    # 计算加权KL散度
    loss = F.kl_div(weighted_prob_s, weighted_prob_t, reduction='batchmean', log_target=False)
    return loss * (temperature ** 2)


def cosine_annealing(current_step, start=0, end=2.5, total_steps=400):
    progress = min(current_step / (total_steps - 1), 1.0)
    return start + (end - start) * (1 - math.cos(math.pi * progress)) / 2

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt
class SoftTarget(nn.Module):
    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T=T
        #self.weight=nn.Parameter(torch.ones(1).to('cuda:0').requires_grad_()) 
    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T#*self.weight
        # print("self.weight:")
        # print(self.weight)
        return loss

#criterionKD = SoftTarget(args.T)#4.0
#kd_loss = criterionKD(out_s, out_t.detach()) * args.lambda_kd#1.0
class ourCrossEntropyLoss(nn.Module):
    def __init__(self,weight):
        super(ourCrossEntropyLoss,self).__init__()
        self.weight=nn.Parameter(torch.ones(1).to('cuda:0').requires_grad_())
    def forward(self,outputs,targets):
        log_softmax_outputs=torch.log_softmax(outputs,dim=1)
        wlog_softmax_outputs=self.weight*log_softmax_outputs
        loss=-torch.sum(targets*wlog_softmax_outputs,dim=1)
        print("self.weight:")
        print(self.weight)
        return torch.mean(loss)
# c_weight=nn.Parameter(torch.ones(1).to('cuda:0').requires_grad_())
# a_weight=nn.Parameter(torch.ones(1).to('cuda:0').requires_grad_())
def end_to_end_finetune(train_loader, test_loader, model, t_model, delete_model,args,flag):
    # Data loading code
    end = time.time()
    if args.FT == 'MiR':
        criterion = torch.nn.MSELoss(reduction='mean').cuda()
    elif args.FT == 'BP':
        criterion = torch.nn.CrossEntropyLoss().cuda()

    # model.fc.requires_grad = False
    #model.freeze_classifier()
    criterionKD = SoftTarget(args.T)#4.0
    criterionKD_d = SoftTarget(args.T)#4.0
    # optimizer = optim.SGD(
    #     [{"params":model.parameters()},{"params":criterionKD.parameters()},{"params":criterionKD_d.parameters()}], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
   
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(0.4 * args.epoch), gamma=0.1)

    # switch to train mode
    model.train()
    #model.get_feat = 'pre_GAP'
    model.get_feat = 'after_GAP'
    delete_model.get_feat='after_GAP'
    delete_model.train()
    t_model.eval()
    #t_model.get_feat = 'pre_GAP'
    t_model.get_feat = 'after_GAP'

    iter_nums = 0
    flag_kl_max=False
    torch.cuda.empty_cache()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    finish = False
    while not finish:
        for batch_idx, (data, target) in enumerate(train_loader):
            iter_nums += 1
            loss=0.0
            loss_d=0.0
            kd_loss=0.0
            kd_loss_d=0.0
            loss_m=0.0
            if iter_nums > args.epoch:
                finish = True
                break
            # measure data loading time
            data = data.cuda()
            target = target.cuda()
            data_time.update(time.time() - end)
            optimizer.zero_grad()
            output, s_features = model(data)
            output_d, d_features = delete_model(data)
            
            a=0.0
            c=0.0
            with torch.no_grad():
                t_output, t_features = t_model(data)
            if args.FT == 'MiR':
                loss = criterion(s_features, t_features)
                #loss_d=
            elif args.FT == 'BP':
                #criterion_mse = torch.nn.MSELoss(reduction='mean').cuda()
                #kd_loss=std_loss(output,t_output,2)
                #kd_loss=dkd_loss(output, t_output, target, 1, 8, 4,True)
                #criterionKD(output,t_output.detach())*args.lambda_kd
                # for i in range(len(output)//2):
                #     kd_loss=kd_loss+criterionKD(output[len(output)-i-1],t_output[len(t_output)-i-1].detach())*args.lambda_kd
                
                # for i in range(1,min(len(output),len(t_output))):
                #     kd_loss=kd_loss+criterionKD(output[i],t_output[i].detach())* args.lambda_kd#1.0
                #kd_loss = criterionKD(output[0], t_output[0].detach()) * args.lambda_kd#1.0
                if flag==True:
                    #pass
                    #$kd_loss_d=dkd_loss(output[0], output_d[0], target, 1, 8, 8,False)
                    #beta
                    kd_loss =dkd_loss(output, t_output.detach(), target, 1, 4.0, 5, iter_nums, flag_kl_max)#std_loss(output,t_output,2)#dkd_loss(output, t_output.detach(), target, 1, 2.0, 5, iter_nums, flag_kl_max)#std_loss(output,t_output,4)#dkd_loss(output, t_output.detach(), target, 1, 4.0, 5, iter_nums, flag_kl_max)#std_loss(output,t_output,4)#dkd_loss(output, t_output.detach(), target, 1, 4.0, 5, iter_nums, flag_kl_max)#std_loss(output,t_output,4) #
                    kd_loss_d=dkd_loss(output,output_d.detach(), target,1, 4.0,5,iter_nums,flag_kl_max)#criterionKD_d(output,output_d.detach())#criterionKD_d(output,output_d.detach())*args.lambda_kd
                    # for i in range(min(len(output)//2,len(output_d))):
                    #     kd_loss_d=kd_loss_d+criterionKD_d(output[i],output_d[i].detach())*args.lambda_kd
                    # for i in range(1,min(len(output),len(output_d))):
                    #    kd_loss_d=kd_loss_d+dkd_loss(output[i], output_d[i], target, 1, 8, 4,False)*args.lambda_kd
                    #kd_loss_d=criterionKD_d(output[0],output_d[0].detach())*args.lambda_kd
                else:
                    kd_loss=criterionKD(output,t_output.detach())#dkd_loss(output, t_output.detach(), target, 1, 4.0, 5, iter_nums, flag_kl_max)#criterionKD(output,t_output.detach())
                loss=loss+criterion(output, target)
                # for ii in range(len(output)):
                #     #loss_d=loss_d+criterion(output_d[ii],target)
                #     loss=loss+criterion(output[ii], target)
                #     #loss_m=loss_m+criterion_mse(output[ii],output_d[ii])
                #print(loss_m)
                if flag==True:#训练最终模型
                    #loss=loss+kd_loss_d#+kd_loss
                    # a=1.0
                    b=1.0
                    loss=loss+kd_loss#+kd_loss_d
                    #loss=loss+c*kd_loss_d+a*kd_loss
                    #loss=loss+kd_loss+kd_loss_d#

                    #loss=loss+(1-(6-2*int(args.rm_blocks))/6)*kd_loss_d+(1+(6-2*int(args.rm_blocks))/6)*kd_loss
                else:
                    loss=loss+kd_loss
                #loss = criterion(output, target)
            losses.update(loss.data.item(), data.size(0))
            loss.backward()
            optimizer.step()
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))
            lr = optimizer.param_groups[0]['lr']
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()#.cpu().detach().numpy()
            if iter_nums % args.print_freq == 0:
                print(a)
                print(c)
                print(float(output.mean(dim=-1,keepdims=True).mean()))
                print('Train: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'LR {lr}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    iter_nums, args.epoch, batch_time=batch_time, lr=lr,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
            if iter_nums % args.eval_freq == 0:
                validate(test_loader, model)
                model.train()
                #model.get_feat ='pre_GAP'
                model.get_feat ='after_GAP'
            scheduler.step()
    validate(test_loader, model)


def validate(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluate mode
    model.eval()
    all_preds = []
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        loss=0.0

        with torch.no_grad():
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)

            if isinstance(output, tuple):
                output = output[0]
            #for ii in range(len(output)):
            #    loss=loss+criterion(output[ii], target)
            loss = criterion(output, target)
        #if(output.shape[0]==32):
            #print('ture')
            #all_preds.append(np.sum(output.data.cpu().numpy(),axis=0))
            #np.append(all_preds,output.data.cpu().numpy())
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        #print(output)
        #print(output.shape)
        #print(target)
        #all_preds=np.sum(all_preds,axis=0)
       # print(all_preds)
        #print(res)
       # print(len(all_preds))
        #print(all_preds.shape)
        #print(all_preds[0].shape)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    #length=len(all_preds)*32
    #all_preds=np.array(all_preds)
    #print(all_preds)
    #all_preds=np.sum(all_preds,axis=0)
    #print(all_preds)
    #print(all_preds/length)
    #print(np.std(all_preds/length))
    #print(length)

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
