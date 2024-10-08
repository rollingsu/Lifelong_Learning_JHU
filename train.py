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
warnings.filterwarnings("ignore", message="No module named 'triton'")
import argparse

from data_utils import ClassBufferManager

def set_parse():
    """
    解析命令行参数并返回参数对象。
    """
    parser = argparse.ArgumentParser()
    # 设置命令行参数
    parser.add_argument("--pretrain", type=str, default=False, help="预训练模型的路径")
    parser.add_argument("--resume", type=str, default=False, help="恢复模型的路径")
    parser.add_argument("--data_dir", type=str, default='D:\\JHU summerproject', help="数据集目录")
    parser.add_argument("--dataset_codes", type=list, default=['Pro'], help="数据集代码")
    # 模型配置参数
    parser.add_argument("--test_mode", default=False, type=bool, help="是否以测试模式运行")
    parser.add_argument("--infer_overlap", default=0.5, type=float, help="滑动窗口推理的重叠度")
    parser.add_argument("--spatial_size", default=(32, 256, 256), type=tuple, help="输入图像的空间尺寸")
    parser.add_argument("--patch_size", default=(4, 16, 16), type=tuple, help="补丁的大小")
    parser.add_argument('--work_dir', type=str, default='./work_dir', help="工作目录")
    parser.add_argument("--clip_ckpt", type=str, default='./config/clip', help="Clip模型检查点")
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="随机翻转增强的概率")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="随机缩放强度增强的概率")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="随机偏移强度增强的概率")
    parser.add_argument('--num_workers', type=int, default=8, help="数据加载器的工作线程数")
    # 分布式训练配置
    parser.add_argument('--dist', dest='dist', type=bool, default=False, help='是否使用分布式训练')
    parser.add_argument('--node_rank', type=int, default=0, help='节点排名')
    parser.add_argument('--init_method', type=str, default="env://", help='分布式训练的初始化方法')
    parser.add_argument('--bucket_cap_mb', type=int, default=25, help='DDP桶的容量（MB），影响梯度通信频率')
    # 关键训练参数
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--warmup_epoch', type=int, default=10, help='预热周期数')
    parser.add_argument('--num_epochs', type=int, default=500, help='总训练周期数')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument("--use_pseudo_label", default=False, type=bool, help='是否使用伪标签')
    parser.add_argument('--ewc_lambda', type=float, default=0.4, help='EWC的lambda参数')
    parser.add_argument("--organ_list", default=['liver', 'right kidney', 'spleen', 'pancreas', 'aorta', 'inferior vena cava', 'right adrenal gland', 'left adrenal gland', 'gallbladder', 'esophagus', 'stomach', 'duodenum', 'left kidney'], type=list)
    args = parser.parse_args()
    return args


class ReplayBuffer:
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

def get_bounding_box(mask):
    """
    计算掩码的边界框。
    返回 (min_z, max_z, min_y, max_y, min_x, max_x)。
    """
    assert mask.ndim == 3, "掩码应为3D"
    pos = torch.nonzero(mask)
    if pos.numel() == 0:
        return None  # 没有对象
    min_z, min_y, min_x = pos.min(dim=0)[0]
    max_z, max_y, max_x = pos.max(dim=0)[0]
    return (min_z.item(), max_z.item(), min_y.item(), max_y.item(), min_x.item(), max_x.item())


def crop_tensor(image, bbox, padding=10):
    """
    根据边界框裁剪图像张量，并可选地添加填充。
    """
    min_z, max_z, min_y, max_y, min_x, max_x = bbox
    # 应用填充，同时确保索引在图像边界内
    min_z = max(min_z - padding, 0)
    max_z = min(max_z + padding, image.size(2))
    min_y = max(min_y - padding, 0)
    max_y = min(max_y + padding, image.size(3))
    min_x = max(min_x - padding, 0)
    max_x = min(max_x + padding, image.size(4))
    return image[:, :, min_z:max_z, min_y:max_y, min_x:max_x]

def inference(model, dataloader, device):
    """
    推理函数，计算平均Dice系数。
    修改为按器官裁剪进行推理。
    """
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for batch in dataloader:
            image, gt3D = batch["image"].to(device), batch["post_label"].to(device)
            organ_name_list = batch['organ_name_list']

            for cls_idx in range(len(organ_name_list)):
                organs_cls = organ_name_list[cls_idx]
                labels_cls = gt3D[:, cls_idx]

                if torch.sum(labels_cls) == 0:
                    continue  # 当前器官不存在，跳过

                # 计算边界框
                bbox = get_bounding_box(labels_cls.squeeze())
                if bbox is None:
                    continue  # 无边界框，跳过

                # 根据边界框裁剪图像和标签
                cropped_image = crop_tensor(image, bbox)  # 形状：[B, C, D, H, W]
                cropped_label = crop_tensor(labels_cls.unsqueeze(1), bbox)  # 形状：[B, 1, D, H, W]

                # 前向传播
                outputs = model(cropped_image, organs=None, boxes=None, points=None, train_organs=organs_cls, train_labels=cropped_label)
                outputs = torch.sigmoid(outputs) > 0.5
                dice = dice_score(outputs, cropped_label.squeeze())
                dice_scores.append(dice)

    model.train()
    if len(dice_scores) == 0:
        return 0.0
    return sum(dice_scores) / len(dice_scores)


class CombinedLoss(nn.Module):
    """
    组合损失函数，结合Dice损失和二元交叉熵损失。
    """
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
    训练一个周期，按器官进行裁剪和分割。
    返回周期的损失、迭代次数和开始时间。
    """
    epoch_loss = 0  
    epoch_iterator = tqdm(train_dataloader, desc=f"[RANK {rank}]", dynamic_ncols=True)  
    
    for batch in epoch_iterator:
        image, gt3D = batch["image"].to(device), batch["post_label"].to(device)
        organ_name_list = batch['organ_name_list'] 

        for cls_idx in range(len(organ_name_list)):
            organs_cls = organ_name_list[cls_idx] 
            labels_cls = gt3D[:, cls_idx] 

            if torch.sum(labels_cls) == 0:
                print(f'[RANK {rank}] ITER-{iter_num} --- 无对象，跳过迭代')
                continue 

            # 计算边界框
            bbox = get_bounding_box(labels_cls.squeeze())
            if bbox is None:
                print(f'[RANK {rank}] ITER-{iter_num} --- 未找到器官 {organs_cls} 的边界框，跳过迭代')
                continue

            # 根据边界框裁剪图像和标签
            cropped_image = crop_tensor(image, bbox)  # 形状：[B, C, D, H, W]
            cropped_label = crop_tensor(labels_cls.unsqueeze(1), bbox)  # 形状：[B, 1, D, H, W]

            optimizer.zero_grad()  

            # 使用裁剪后的数据进行前向传播
            outputs = model(cropped_image, organs=None, boxes=None, points=None, train_organs=organs_cls, train_labels=cropped_label)

            # 计算损失
            loss = ewc_loss_function(outputs, cropped_label.squeeze(), model)
            loss.backward()  
            optimizer.step() 

            print(f'[RANK {rank}] ITER-{iter_num} --- 器官: {organs_cls}, 损失: {loss.item()}')
            iter_num += 1 

            if replay_buffer is not None:
                replay_buffer.add((cropped_image, cropped_label))

            # 记录日志
            if rank == 0:
                args.writer.add_scalar('train_iter/loss', loss.item(), iter_num)

            epoch_loss += loss.item()

    scheduler.step() 
    epoch_loss /= len(train_dataloader) + 1e-12
    print(f'{args.model_save_path} ==> [RANK {rank}] epoch_loss: {epoch_loss}')

    if rank == 0:
        args.writer.add_scalar('train/loss', epoch_loss, epoch)
        args.writer.add_scalar('train/lr', scheduler.get_lr(), epoch)

    if datetime.now() >= start_time + timedelta(days=3):
        dice = inference(model, train_dataloader, device)
        print(f'[RANK {rank}] 3天后的DICE: {dice:.4f}')
        start_time = datetime.now()

    return epoch_loss, iter_num, start_time

# parameters comes from
def compute_fisher(model, criterion, dataloader, device):
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
    for name in fisher:
        fisher[name] /= len(dataloader)
    return fisher

def main():

    """
    主函数，负责初始化和启动训练过程。
    """
    torch.manual_seed(2025)
    torch.cuda.empty_cache()
    args = set_parse() 
    
    os.environ["TOKENIZERS_PARALLELISM"] = "False"  
    args.run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = os.path.join(args.work_dir, args.run_id)
    args.model_save_path = model_save_path

    ngpus_per_node = torch.cuda.device_count()
    print("每节点GPU数量={}".format(ngpus_per_node))
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
    """
    每个进程的主工作函数，负责模型的初始化、训练和保存。
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    rank = gpu
    world_size = ngpus_per_node
    print(f"[Rank {rank}]: 使用设备: {device} 进行训练")
    is_main_host = rank == 0
    
    buffer_manager = ClassBufferManager(buffer_path=os.path.join(args.model_save_path, 'class_buffer.json'), max_classes=20)

    if is_main_host:
        os.makedirs(args.model_save_path, exist_ok=True)
        shutil.copyfile(__file__, os.path.join(args.model_save_path, args.run_id + '_' + os.path.basename(__file__)))

    sam_model = sam_model_registry['vit'](args=args, checkpoint=None)

    if args.pretrain and os.path.isfile(args.pretrain):
            print(f'[Rank {rank}] 加载预训练模型: {args.pretrain}')
            model = lifelong(
                image_encoder=UNet(
                    spatial_dims=3, 
                    in_channels=1, 
                    out_channels=1, 
                    channels=(32, 64, 128, 256, 512),      
                    strides=(2, 2, 2, 2),
                    text_embedding_dim= 512,
                    clip_ckpt=args.clip_ckpt
                ),
                mask_decoder=sam_model.mask_decoder,
                clip_ckpt=args.clip_ckpt,
                roi_size=args.spatial_size,
                patch_size=args.patch_size,
                test_mode=args.test_mode,
            ).to(device)
            checkpoint = torch.load(args.pretrain, map_location=device)
            model.load_state_dict(checkpoint['model'])
            print(f'[Rank {rank}] pretriand model loaded')
    else:
            print(f'[Rank {rank}] vanilla model')
            model = lifelong(
                image_encoder=UNet(
                    spatial_dims=3, 
                    in_channels=1, 
                    out_channels=1, 
                    channels=(32, 64, 128, 256, 512),      
                    strides=(2, 2, 2, 2),
                    text_embedding_dim=512,
                    clip_ckpt=args.clip_ckpt
                ),
                clip_ckpt=args.clip_ckpt,
                roi_size=args.spatial_size,
                patch_size=args.patch_size,
                test_mode=args.test_mode,
                prompt_encoder=sam_model.prompt_encoder,
                mask_decoder=sam_model.mask_decoder,
            ).to(device)

    if ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method=args.init_method,
            rank=rank,
            world_size=world_size,
        )
        print('初始化进程组完成')

    sam_model = sam_model_registry['vit'](args=args, checkpoint=None)
    model = lifelong(
                        image_encoder=UNet(
                            spatial_dims=3, 
                            in_channels=1, 
                            out_channels=1, 
                            channels=(32, 64, 128, 256, 512),      
                            strides=(2, 2, 2, 2),
                            text_embedding_dim=512,
                            clip_ckpt=args.clip_ckpt
                        ),      
                        # image_encoder=sam_model.image_encoder, 
                        mask_decoder=sam_model.mask_decoder,
                        clip_ckpt=args.clip_ckpt,
                        roi_size=args.spatial_size,
                        patch_size=args.patch_size,
                        test_mode=args.test_mode,
                        prompt_encoder=sam_model.prompt_encoder,
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

    # 定义优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 定义学习率调度器
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.num_epochs)
    replay_buffer = ReplayBuffer(1000) 

    num_epochs = args.num_epochs
    iter_num = 0
    start_time = datetime.now()  

    train_dataloader = get_loader(args, buffer_manager)
    start_epoch = 0

    # 加载检查点
    if args.resume is not None and os.path.isfile(args.resume):
        print(rank, "=> 正在加载检查点 '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        scheduler.last_epoch = start_epoch
        print(rank, "=> 加载检查点 '{}' (周期 {})".format(args.resume, checkpoint['epoch']))
        if ngpus_per_node > 1:
            torch.distributed.barrier()

    # 初始化TensorBoard日志
    if rank == 0:
        args.writer = SummaryWriter(log_dir='./tb_log/' + args.run_id)
        print('正在写入Tensorboard日志到 ', './tb_log/' + args.run_id)

    # 定义基础损失函数
    base_loss = CombinedLoss()

    # 初始化Fisher信息和参数字典
    fisher_dict = {}
    params_dict = {}

    # 定义EWC损失函数
    ewc_loss_function = EWCLoss(criterion=base_loss, ewc_lambda=args.ewc_lambda, fisher_dict=fisher_dict, params_dict=params_dict)

    for epoch in range(start_epoch, num_epochs):
        if ngpus_per_node > 1:
            with model.join():
                epoch_loss, iter_num, start_time = train_epoch(
                    args, model, train_dataloader, optimizer, scheduler, epoch, rank, device, 
                    iter_num, start_time, ewc_loss_function, fisher_dict, params_dict, replay_buffer
                )
        else:
            epoch_loss, iter_num, start_time = train_epoch(
                args, model, train_dataloader, optimizer, scheduler, epoch, rank, device, 
                iter_num, start_time, ewc_loss_function, fisher_dict, params_dict, replay_buffer
            )

        print(f'时间: {datetime.now().strftime("%Y%m%d-%H%M")}, 周期: {epoch}, 损失: {epoch_loss}')

        # 保存模型检查点
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
            print('正在计算 Fisher 信息矩阵...')
            fisher = compute_fisher(model, base_loss, train_dataloader, device)
            for name, param in model.named_parameters():
                params_dict[name] = param.data.clone()
                fisher_dict[name] = fisher[name]
            ewc_loss_function.fisher_dict = fisher_dict
            ewc_loss_function.params_dict = params_dict

    # 保存EWC参数
    if is_main_host:
        torch.save({'fisher_dict': fisher_dict, 'params_dict': params_dict}, os.path.join(args.model_save_path, 'ewc_params.pth'))
        buffer_manager.save()
    train_dataloader = get_loader(args, buffer_manager)


if __name__ == "__main__":
    main()
