'''
@File       :   train.py
@Time       :   2023/02/04 10:51:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   Train reward model.
'''

# LOCAL
from models.src.config.options import *
from models.src.config.utils import *
from models.src.config.learning_rates import get_learning_rate_scheduler
os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
opts.BatchSize = opts.batch_size * opts.accumulation_steps * opts.gpu_num

from dataset import DiversityDataset
from models.src.ImageReward import ImageReward
from utils.parser import Parser
from models.baseline_clip import preprocess

# GLOBAL
import torch
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.backends import cudnn
from sklearn.model_selection import train_test_split


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


def loss_func(predict, target):
    loss_list = F.binary_cross_entropy(predict, target, reduction='none')
    loss = torch.mean(loss_list)
    labels_pred = (predict > 0.5) * 1.0
    acc = (labels_pred == target).sum().clone().detach().float()
    
    return loss, loss_list, acc


if __name__ == "__main__":
    
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

    parser = Parser()
    paths = ['../files/0_500_pickscore_coco',
             '../files/diverse_coco_pick_3_per_prompt_500_1000.out',
             '../files/diverse_coco_pick_3_per_prompt_1000_1500',
             '../files/diverse_coco_pick_3_per_prompt_1500_2000',
             '../files/diverse_coco_pick_3_per_prompt_2000_2500']
    df = parser.raw_to_df(paths, do_overlap=True, keep_no_info=False)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

    train_dataset = DiversityDataset(train_df,
                                     local_path='/extra_disk_1/quickjkee/diversity_images',
                                     preprocess=None)
    valid_dataset = DiversityDataset(test_df,
                                     local_path='/extra_disk_1/quickjkee/diversity_images',
                                     preprocess=None)
    test_dataset = valid_dataset
    
    if opts.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, sampler=train_sampler, collate_fn=collate_fn if not opts.rank_pair else None)
    else:
        train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn if not opts.rank_pair else None)
    
    valid_loader = DataLoader(valid_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn if not opts.rank_pair else None)
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn if not opts.rank_pair else None)

    # Set the training iterations.
    opts.train_iters = opts.epochs * len(train_loader)
    steps_per_valid = len(train_loader) // opts.valid_per_epoch
    print("len(train_dataset) = ", len(train_dataset))
    print("train_dataset.iters_per_epoch = ", train_dataset.iters_per_epoch)
    print("len(train_loader) = ", len(train_loader))
    print("steps_per_valid = ", steps_per_valid)

    model = ImageReward(device).to(device)
    
    if opts.preload_path:
        model = preload_model(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.adam_beta1, opts.adam_beta2), eps=opts.adam_eps)
    scheduler = get_learning_rate_scheduler(optimizer, opts)
    if opts.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)


    # valid result print and log
    if get_rank() == 0:
        model.eval()
        valid_loss = []
        valid_acc_list = []
        with torch.no_grad():
            for step, batch_data_package in enumerate(valid_loader):
                predict = model(batch_data_package)
                target = batch_data_package['background'].to(device)
                loss, loss_list, acc = loss_func(predict, target)
                valid_loss.append(loss_list)
                valid_acc_list.append(acc.item())
    
        # record valid and save best model
        valid_loss = torch.cat(valid_loss, 0)
        print('Validation - Iteration %d | Loss %6.5f | Acc %6.4f' % (0, torch.mean(valid_loss), sum(valid_acc_list) / len(valid_acc_list)))
        writer.add_scalar('Validation-Loss', torch.mean(valid_loss), global_step=0)
        writer.add_scalar('Validation-Acc', sum(valid_acc_list) / len(valid_acc_list), global_step=0)
            

    best_loss = 1e9
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
            target = batch_data_package['background'].to(device)
            loss, loss_list, acc = loss_func(predict, target)
            # loss regularization
            loss = loss / opts.accumulation_steps
            # back propagation
            loss.backward()

            losses.append(loss_list)
            acc_list.append(acc.item())

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
                    print('Iteration %d | Loss %6.5f | Acc %6.4f' % (train_iteration, torch.mean(losses_log), sum(acc_list) / len(acc_list)))
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
                    with torch.no_grad():
                        for step, batch_data_package in enumerate(valid_loader):
                            predict = model(batch_data_package)
                            target = batch_data_package['background'].to(device)
                            loss, loss_list, acc = loss_func(predict, target)
                            valid_loss.append(loss_list)
                            valid_acc_list.append(acc.item())
                
                    # record valid and save best model
                    valid_loss = torch.cat(valid_loss, 0)
                    print('Validation - Iteration %d | Loss %6.5f | Acc %6.4f' % (train_iteration, torch.mean(valid_loss), sum(valid_acc_list) / len(valid_acc_list)))
                    writer.add_scalar('Validation-Loss', torch.mean(valid_loss), global_step=train_iteration)
                    writer.add_scalar('Validation-Acc', sum(valid_acc_list) / len(valid_acc_list), global_step=train_iteration)
                        
                    if torch.mean(valid_loss) < best_loss:
                        print("Best Val loss so far. Saving model")
                        best_loss = torch.mean(valid_loss)
                        print("best_loss = ", best_loss)
                        save_model(model)

    # test model
    if get_rank() == 0:
        print("training done")
        print("test: ")
        model = load_model(model)
        model.eval()

        test_loss = []
        acc_list = []
        with torch.no_grad():
            for step, batch_data_package in enumerate(test_loader):
                reward = model(batch_data_package)
                loss, loss_list, acc = loss_func(reward)
                test_loss.append(loss_list)
                acc_list.append(acc.item())

        test_loss = torch.cat(test_loss, 0)
        print('Test Loss %6.5f | Acc %6.4f' % (torch.mean(test_loss), sum(acc_list) / len(acc_list)))

