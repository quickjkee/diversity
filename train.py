'''
@File       :   train.py
@Time       :   2023/02/04 10:51:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   Train reward model.
'''
import sys

sys.path.append(f'/home/quickjkee/diversity/models/src/config')
sys.path.append(f'/home/quickjkee/diversity/models/src')
sys.path.append(f'/home/quickjkee/diversity/models')

# LOCAL
from models.src.config.options import *
from models.src.config.utils import *
from models.src.config.learning_rates import get_learning_rate_scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
opts.BatchSize = opts.batch_size * opts.accumulation_steps * opts.gpu_num

# GLOBAL
import torch
import math
import numpy as np
import torch.nn as nn
import sys
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler, Sampler
from torch.backends import cudnn
from metrics import samples_metric
from catalyst.data.sampler import DistributedSamplerWrapper


def std_log():
    if get_rank() == 0:
        save_path = make_path()
        makedir(config['log_base'])
        sys.stdout = open(os.path.join(config['log_base'], "{}.txt".format(save_path)), "w")


def init_seeds(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        print(alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss, ce_loss


def run_train(train_dataset,
              valid_dataset,
              model,
              label,
              loss_w):

    if opts.std_log:
        std_log()

    if opts.distributed:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        init_seeds(opts.seed + local_rank)

    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        init_seeds(opts.seed)

    writer = visualizer()
    model.to(device)
    model.device = device
    loss_fn = FocalLoss(alpha=loss_w.to(device) * 500)
    test_dataset = valid_dataset

    def loss_func(predict, target):
        loss, loss_list = loss_fn(predict, target)
        preds_probs = F.softmax(predict, dim=1)
        correct = torch.eq(torch.max(preds_probs, dim=1)[1], target).view(-1)
        acc = torch.sum(correct).item() / len(target)

        return loss, loss_list, acc

    #def loss_func(predict, target):
    #    loss = nn.CrossEntropyLoss(reduction='none')
    #    loss_list = loss(predict, target)
    #    loss = torch.mean(loss_list, dim=0)
    #    correct = torch.eq(torch.max(F.softmax(predict, dim=1), dim=1)[1], target).view(-1)
    #    acc = torch.sum(correct).item() / len(target)
    #    return loss, loss_list, acc

    if opts.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, sampler=train_sampler,
                                  collate_fn=collate_fn if not opts.rank_pair else None)
    else:
        train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True,
                                  collate_fn=collate_fn if not opts.rank_pair else None)

    valid_loader = DataLoader(valid_dataset, batch_size=opts.batch_size, shuffle=True,
                              collate_fn=collate_fn if not opts.rank_pair else None)
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True,
                             collate_fn=collate_fn if not opts.rank_pair else None)

    # Set the training iterations.
    opts.train_iters = opts.epochs * len(train_loader)
    steps_per_valid = len(train_loader) // opts.valid_per_epoch
    print("len(train_dataset) = ", len(train_dataset))
    print("len(train_loader) = ", len(train_loader))
    print("len(test_dataset) = ", len(test_dataset))
    print("len(test_loader) = ", len(test_loader))
    print("steps_per_valid = ", steps_per_valid)

    if opts.preload_path:
        model = preload_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.adam_beta1, opts.adam_beta2),
                                 eps=opts.adam_eps)
    scheduler = get_learning_rate_scheduler(optimizer, opts)
    if opts.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)

    # valid result print and log
    if get_rank() == 0:
        model.eval()
        valid_loss = []
        valid_acc_list = []
        labels_preds = []
        labels_true = []
        with torch.no_grad():
            for step, batch_data_package in enumerate(valid_loader):
                predict = model(batch_data_package)
                target = batch_data_package[label].to(device)
                loss, loss_list, acc = loss_func(predict, target)

                labels_preds.append(torch.max(F.softmax(predict, dim=1), dim=1)[1].cpu().view(-1).int())
                labels_true.append(target.cpu().view(-1).int())
                valid_loss.append(loss_list)
                valid_acc_list.append(acc)

        # record valid and save best model
        valid_loss = torch.cat(valid_loss, 0)
        labels_preds = list(np.array(torch.cat(labels_preds, 0)))
        labels_true = list(np.array(torch.cat(labels_true, 0)))
        boots_acc = samples_metric(labels_true, labels_preds)[0]
        print('Validation - Iteration %d | Loss %6.5f | Acc %6.4f | BootsAcc %6.4f' % (
        0, torch.mean(valid_loss), sum(valid_acc_list) / len(valid_acc_list), boots_acc))
        writer.add_scalar('Validation-Loss', torch.mean(valid_loss), global_step=0)
        writer.add_scalar('Validation-Acc', sum(valid_acc_list) / len(valid_acc_list), global_step=0)
        writer.add_scalar('Validation-BootsAcc', boots_acc, global_step=0)

    best_acc = 0
    optimizer.zero_grad()
    # fix_rate_list = [float(i) / 10 for i in reversed(range(10))]
    # fix_epoch_edge = [opts.epochs / (len(fix_rate_list)+1) * i for i in range(1, len(fix_rate_list)+1)]
    # fix_rate_idx = 0
    losses = []
    acc_list = []
    for epoch in range(opts.epochs):

        for step, batch_data_package in enumerate(train_loader):
            model.train()
            predict = model(batch_data_package)
            target = batch_data_package[label].to(device)
#            print(target)
            loss, loss_list, acc = loss_func(predict, target)
            # loss regularization
            loss = loss / opts.accumulation_steps
            # back propagation
            loss.backward()

            losses.append(loss_list)
            acc_list.append(acc)

            iterations = epoch * len(train_loader) + step + 1
            train_iteration = iterations / opts.accumulation_steps

            # update parameters of net
            if (iterations % opts.accumulation_steps) == 0:
                # optimizer the net
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                # train result print and log 
                if get_rank() == 0:
                    losses_log = torch.cat(losses, 0)
                    print('Iteration %d | Loss %6.5f | Acc %6.4f' % (
                    train_iteration, torch.mean(losses_log), sum(acc_list) / len(acc_list)))
                    writer.add_scalar('Train-Loss', torch.mean(losses_log), global_step=train_iteration)
                    writer.add_scalar('Train-Acc', sum(acc_list) / len(acc_list), global_step=train_iteration)

                losses.clear()
                acc_list.clear()

            # valid result print and log
            if (iterations % steps_per_valid) == 0:
                if get_rank() == 0:
                    model.eval()
                    valid_loss = []
                    valid_acc_list = []
                    labels_preds = []
                    labels_true = []
                    with torch.no_grad():
                        for step, batch_data_package in enumerate(valid_loader):
                            predict = model(batch_data_package)
                            target = batch_data_package[label].to(device)
                            loss, loss_list, acc = loss_func(predict, target)

                            labels_preds.append(torch.max(F.softmax(predict, dim=1), dim=1)[1].cpu().view(-1).int())
                            labels_true.append(target.cpu().view(-1).int())
                            valid_loss.append(loss_list)
                            valid_acc_list.append(acc)

                    # record valid and save best model
                    valid_loss = torch.cat(valid_loss, 0)
                    labels_preds = list(np.array(torch.cat(labels_preds, 0)))
                    labels_true = list(np.array(torch.cat(labels_true, 0)))
                    boots_acc = samples_metric(labels_true, labels_preds)[0]
                    print('Validation - Iteration %d | Loss %6.5f | Acc %6.4f | BootsAcc %6.4f' % (
                    train_iteration, torch.mean(valid_loss), sum(valid_acc_list) / len(valid_acc_list), boots_acc))
                    writer.add_scalar('Validation-Loss', torch.mean(valid_loss), global_step=train_iteration)
                    writer.add_scalar('Validation-Acc', sum(valid_acc_list) / len(valid_acc_list),
                                      global_step=train_iteration)
                    writer.add_scalar('Validation-BootsAcc', boots_acc, global_step=train_iteration)

                    if boots_acc > best_acc:
                        print("Best BootsAcc so far. Saving model")
                        best_acc = boots_acc
                        print("best_acc = ", best_acc)
                        save_model(model)

    # test model
    if get_rank() == 0:
        print("training done")
        print("test: ")
        model = load_model(model)
        model.eval()

        test_loss = []
        acc_list = []
        labels_preds = []
        labels_true = []
        with torch.no_grad():
            for step, batch_data_package in enumerate(test_loader):
                predict = model(batch_data_package)
                target = batch_data_package[label].to(device)
                loss, loss_list, acc = loss_func(predict, target)

                labels_preds.append(torch.max(F.softmax(predict, dim=1), dim=1)[1].cpu().view(-1))
                labels_true.append(target.cpu().view(-1))
                test_loss.append(loss_list)
                acc_list.append(acc)

        test_loss = torch.cat(test_loss, 0)
        labels_preds = np.array(torch.cat(labels_preds, 0))
        labels_true = np.array(torch.cat(labels_true, 0))
        boots_acc = samples_metric(labels_true, labels_preds)[0]
        print('Test Loss %6.5f | Acc %6.4f' % (torch.mean(test_loss), boots_acc))
