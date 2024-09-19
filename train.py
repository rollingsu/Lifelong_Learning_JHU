from __future__ import absolute_import, division, print_function
import os
import torch
import torch.nn as nn
import argparse
from datetime import datetime, timedelta
import shutil
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.loss import BCELoss, BinaryDiceLoss
from data_utils import get_loader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import warnings
import logging
from network.models import lifelong
from EWC.ewc_loss import EWCLoss
from network.models import UNet
import torch.multiprocessing as mp

from network.models import lifelong  
from network.models import UNet 
from segment_anything_volumetric import sam_model_registry 

import warnings
from cryptography.utils import CryptographyDeprecationWarning

warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import argparse

def set_parse():
    """
    Parse command-line arguments and return the arguments object.
    """
    parser = argparse.ArgumentParser()
    # Set command-line arguments
    parser.add_argument("--pretrain", type=str, default='', help="Path to the pretrained model")
    parser.add_argument("--resume", type=str, default='', help="Path to resume the model")
    parser.add_argument("--data_dir", type=str, default='D:\\JHU summerproject', help="Dataset directory")
    parser.add_argument("--dataset_codes", type=list, default=['0000'], help="Dataset codes")
    # Model configuration parameters
    parser.add_argument("--test_mode", default=False, type=bool, help="Whether to run in test mode")
    parser.add_argument("-infer_overlap", default=0.5, type=float, help="Overlap for sliding window inference")
    parser.add_argument("-spatial_size", default=(32, 256, 256), type=tuple, help="Spatial size of input images")
    parser.add_argument("-patch_size", default=(4, 16, 16), type=tuple, help="Size of patches")
    parser.add_argument('-work_dir', type=str, default='./work_dir', help="Working directory")
    parser.add_argument("--clip_ckpt", type=str, default='./config/clip', help="Clip model checkpoint")
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="Probability of random flip augmentation")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="Probability of random scale intensity augmentation")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="Probability of random shift intensity augmentation")
    parser.add_argument('-num_workers', type=int, default=8, help="Number of worker threads for data loader")
    # Distributed training configuration
    parser.add_argument('--dist', dest='dist', type=bool, default=False, help='Whether to use distributed training')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--init_method', type=str, default="env://", help='Initialization method for distributed training')
    parser.add_argument('--bucket_cap_mb', type=int, default=25, help='Capacity of DDP bucket (MB), affects gradient communication frequency')
    # Key training parameters
    parser.add_argument('-lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('-warmup_epoch', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('-num_epochs', type=int, default=500, help='Total number of training epochs')
    parser.add_argument('-batch_size', type=int, default=1, help='Batch size')
    parser.add_argument("--use_pseudo_label", default=False, type=bool, help='Whether to use pseudo labels')
    parser.add_argument('--ewc_lambda', type=float, default=0.4, help='Lambda parameter for EWC')
    args = parser.parse_args()
    return args


class ReplayBuffer:
    """
    回放缓冲区，用于存储和采样经验数据
    """
    def __init__(self, capacity):
        self.capacity = capacity 
        self.buffer = []  
        self.index = 0  

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.index] = experience  
        self.index = (self.index + 1) % self.capacity  

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

def dice_score(preds, labels):
    smooth = 1e-6
    intersection = torch.sum(preds * labels)
    union = torch.sum(preds) + torch.sum(labels)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

def inference(model, dataloader, device):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for batch in dataloader:
            image, gt3D = batch["image"].to(device), batch["post_label"].to(device)
            output = model(image)
            output = torch.sigmoid(output) > 0.5
            dice = dice_score(output, gt3D)
            dice_scores.append(dice)
    model.train()
    return sum(dice_scores) / len(dice_scores)

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.dice_loss = BinaryDiceLoss()
        self.bce_loss = BCELoss()

    def forward(self, outputs, labels):
        sl_loss_dice = self.dice_loss(outputs.squeeze().float(), labels.squeeze().float())
        sl_loss_bce = self.bce_loss(outputs.squeeze().float(), labels.squeeze().float())
        return sl_loss_dice + sl_loss_bce

def train_epoch(args, model, train_dataloader, optimizer, scheduler, epoch, rank, device, iter_num, start_time, ewc_loss_function, fisher_dict, params_dict, replay_buffer):
    """
    训练一个epoch，返回epoch的损失
    """
    epoch_loss = 0  
    epoch_iterator = tqdm(train_dataloader, desc=f"[RANK {rank}]", dynamic_ncols=True)  
    
    for batch in epoch_iterator:
        image, gt3D = batch["image"].to(device), batch["post_label"].to(device)
        organ_name_list = batch['organ_name_list'] 

        loss_step_avg = 0  

        for cls_idx in range(len(organ_name_list)):
            optimizer.zero_grad()  
            organs_cls = organ_name_list[cls_idx] 
            labels_cls = gt3D[:, cls_idx] 

            if torch.sum(labels_cls) == 0:
                print(f'[RANK {rank}] ITER-{iter_num} --- No object, skip iter')
                continue 

            # 前向传播
            outputs = model(image, organs=None, boxes=None, points=None, train_organs=organs_cls, train_labels=labels_cls)

            # 计算损失
            loss = ewc_loss_function(outputs, labels_cls, model)
            loss_step_avg += loss.item()
            
            loss.backward()  
            optimizer.step() 
            print(f'[RANK {rank}] ITER-{iter_num} --- loss {loss.item()}')
            iter_num += 1 

            if replay_buffer is not None:
                replay_buffer.add((image, gt3D)) 

        loss_step_avg /= len(organ_name_list)  

        print(f'[RANK {rank}] AVG loss {loss_step_avg}')
        if rank == 0:
            args.writer.add_scalar('train_iter/loss', loss_step_avg, iter_num)

        epoch_loss += loss_step_avg

    scheduler.step() 
    epoch_loss /= len(train_dataloader) + 1e-12
    print(f'{args.model_save_path} ==> [RANK {rank}] epoch_loss: {epoch_loss}')

    if rank == 0:
        args.writer.add_scalar('train/loss', epoch_loss, epoch)
        args.writer.add_scalar('train/lr', scheduler.get_lr(), epoch)

    if datetime.now() >= start_time + timedelta(days=3):
        dice = inference(model, train_dataloader, device)
        print(f'[RANK {rank}] DICE after 3 days: {dice:.4f}')
        start_time = datetime.now()

    return epoch_loss, iter_num, start_time

def compute_fisher(model, criterion, dataloader, device):
    # 初始化 Fisher 信息矩阵
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)

    model.eval()
    for batch in dataloader:
        model.zero_grad()
        image, gt3D = batch["image"].to(device), batch["post_label"].to(device)
        outputs = model(image)
        loss = criterion(outputs, gt3D)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data.clone().pow(2)
    # 对 Fisher 信息矩阵取平均
    for name in fisher:
        fisher[name] /= len(dataloader)
    return fisher

def main():
    torch.manual_seed(2025)
    torch.cuda.empty_cache()
    args = set_parse() 
    
    os.environ["TOKENIZERS_PARALLELISM"] = "False"  
    args.run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = os.path.join(args.work_dir, args.run_id)
    args.model_save_path = model_save_path

    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node={}".format(ngpus_per_node))
    print(f"项目保存路径 {args.model_save_path}")

    if ngpus_per_node == 0:
        # 没有可用的GPU，使用单进程训练
        main_worker(0, 1, args)
    else:
        # 使用多进程进行分布式训练
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12345'
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

def main_worker(gpu, ngpus_per_node, args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    rank = gpu
    world_size = ngpus_per_node
    print(f"[Rank {rank}]: Use device: {device} for training")
    is_main_host = rank == 0

    if is_main_host:
        os.makedirs(args.model_save_path, exist_ok=True)
        shutil.copyfile(__file__, os.path.join(args.model_save_path, args.run_id + '_' + os.path.basename(__file__)))

    if ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method=args.init_method,
            rank=rank,
            world_size=world_size,
        )
        print('init_process_group finished')


    sam_model = sam_model_registry['vit'](args=args, checkpoint=None)
    model =  lifelong(
                        image_encoder= UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2)),      
                        # image_encoder=sam_model.image_encoder, 
                        mask_decoder=sam_model.mask_decoder,
                        prompt_encoder=sam_model.prompt_encoder,
                        clip_ckpt=args.clip_ckpt,
                        roi_size=args.spatial_size,
                        patch_size=args.patch_size,
                        test_mode=args.test_mode,
                        ).to(device)

    if ngpus_per_node > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[gpu],
            output_device=gpu,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
            bucket_cap_mb=args.bucket_cap_mb
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.num_epochs)
    replay_buffer = ReplayBuffer(1000) 

    num_epochs = args.num_epochs
    iter_num = 0
    start_time = datetime.now()  
    train_dataloader = get_loader(args)
    start_epoch = 0

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(rank, "=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']
            scheduler.last_epoch = start_epoch
            print(rank, "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        if ngpus_per_node > 1:
            torch.distributed.barrier()

    if rank == 0:
        args.writer = SummaryWriter(log_dir='./tb_log/' + args.run_id)
        print('Writing Tensorboard logs to ', './tb_log/' + args.run_id)

    base_loss = CombinedLoss()

    fisher_dict = {}
    params_dict = {}

    ewc_loss_function = EWCLoss(criterion=base_loss, ewc_lambda=args.ewc_lambda, fisher_dict=fisher_dict, params_dict=params_dict)

    for epoch in range(start_epoch, num_epochs):
        if ngpus_per_node > 1:
            with model.join():
                epoch_loss, iter_num, start_time = train_epoch(args, model, train_dataloader, optimizer, scheduler, epoch, rank, device, iter_num, start_time, ewc_loss_function, fisher_dict, params_dict, replay_buffer)
        else:
            epoch_loss, iter_num, start_time = train_epoch(args, model, train_dataloader, optimizer, scheduler, epoch, rank, device, iter_num, start_time, ewc_loss_function, fisher_dict, params_dict, replay_buffer)

        print(f'时间: {datetime.now().strftime("%Y%m%d-%H%M")}, 周期: {epoch}, 损失: {epoch_loss}')

        if is_main_host and (epoch + 1) % 10 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.model_save_path, f'model_epoch{epoch + 1}.pth'))
        if ngpus_per_node > 1:
            torch.distributed.barrier()

        # 在每个任务结束后计算 Fisher 信息并保存模型参数
        if (epoch + 1) % args.num_epochs == 0:
            print('计算 Fisher 信息矩阵...')
            fisher = compute_fisher(model, base_loss, train_dataloader, device)
            for name, param in model.named_parameters():
                params_dict[name] = param.data.clone()
                fisher_dict[name] = fisher[name]
            ewc_loss_function.fisher_dict = fisher_dict
            ewc_loss_function.params_dict = params_dict

    if is_main_host:
        torch.save({'fisher_dict': fisher_dict, 'params_dict': params_dict}, os.path.join(args.model_save_path, 'ewc_params.pth'))

if __name__ == "__main__":
    main()
