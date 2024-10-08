import math
import os
import numpy as np
import torch
from monai import data, transforms
import itertools
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, ConcatDataset
import os
import ast
from scipy import sparse
import random
from scipy.ndimage import binary_opening, binary_closing
from scipy.ndimage import label as label_structure
from scipy.ndimage import sum as sum_structure
import json
from network.models import TextEncoder
import SimpleITK as sitk
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextConfig
import json
import os
import argparse

def set_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--organ_list", default=['liver', 'right kidney', 'spleen', 'pancreas', 'aorta', 'inferior vena cava', 'right adrenal gland', 'left adrenal gland', 'gallbladder', 'esophagus', 'stomach', 'duodenum', 'left kidney'], type=list)
    parser.add_argument("-image_dir", type=str, required=True)
    parser.add_argument("-label_dir", type=str, required=True)
    parser.add_argument("-dataset_code", type=str, required=True)
    parser.add_argument("-save_root", type=str, required=True)
    parser.add_argument("-test_ratio", type=float, required=True)
    args = parser.parse_args()
    return args

# 类定义：ClassBufferManager负责管理类别及其对应的损失，采用缓冲区机制保留重要类别
class ClassBufferManager:
    def __init__(self, buffer_path, max_classes=20):
        # 初始化缓冲区的路径和最大类别数
        self.buffer_path = buffer_path
        self.max_classes = max_classes
        self.classes = {}  # 存储类别名称到对应损失的映射

        # 如果缓冲区路径存在，则从JSON文件中加载已有类别信息
        if os.path.exists(self.buffer_path):
            with open(self.buffer_path, 'r') as f:
                self.classes = json.load(f)
        else:
            # 初始化为空的类别字典
            self.classes = {}

    def update_class_loss(self, class_name, loss):
        """更新指定类别的损失值"""
        self.classes[class_name] = loss
        self._prune_classes()  # 更新完损失后修剪缓冲区，保留损失最大的类别

    def add_new_class(self, class_name, initial_loss=0.0):
        """添加新类别到缓冲区"""
        if class_name not in self.classes:
            self.classes[class_name] = initial_loss
            self._prune_classes()  # 添加新类别后修剪缓冲区

    def _prune_classes(self):
        """保留损失最大的前max_classes个类别"""
        if len(self.classes) > self.max_classes:
            # 将类别按照损失值降序排列，保留损失最大的前max_classes个类别
            sorted_classes = sorted(self.classes.items(), key=lambda item: item[1], reverse=True)
            self.classes = dict(sorted_classes[:self.max_classes])

    def get_active_classes(self):
        """获取当前活跃的类别列表"""
        return list(self.classes.keys())

    def save(self):
        """保存缓冲区到JSON文件"""
        with open(self.buffer_path, 'w') as f:
            json.dump(self.classes, f, indent=4)  # 将类别信息保存为JSON格式

# 数据集拼接：将多个数据集组合成一个数据集
class UnionDataset(Dataset):
    def __init__(self, concat_dataset, datasets):
        self.datasets = datasets  # 保存每个子数据集
        self.lengths = [len(d) for d in datasets]  # 每个数据集的长度
        self.offsets = torch.cumsum(torch.tensor([0] + self.lengths), dim=0)  # 计算偏移量
        self.concat_dataset = concat_dataset  # 合并后的数据集
        
    def __len__(self):
        # 返回所有数据集的总长度
        return sum(self.lengths)

    def __getitem__(self, idx):
        # 根据索引返回数据
        return self.concat_dataset[idx]
    
# 通用数据集：用于处理不同器官分割的数据集，能够结合缓冲区管理机制
class UniversalDataset(Dataset):
    def __init__(self, data, transform, test_mode, organ_list, buffer_manager, clip_ckpt): 
       self.data = data
       self.transform = transform
       self.num_positive_extra_max = 10
       self.num_negative_extra_max = 10
       self.test_mode = test_mode
       self.bbox_shift = 10 if test_mode else 0
       print(organ_list)
       self.target_list = organ_list
       self.text_encoder = TextEncoder(clip_ckpt)  # 使用TextEncoder进行文本编码
       self.ipt_classes = buffer_manager.get_active_classes()  # 从缓冲区中获取活跃类别
       print(f"Buffer classes: {self.ipt_classes}")

       self.active_samples = []  # 用于存储活跃的样本
       for item in self.data:
            image_path = item['image']
            label_path = item['label']
            gt_array = self.load_nifti_image(label_path)  # 加载标签文件
            # 筛选出在当前数据中存在的类别，加入活跃样本列表
            present_classes = [cls for cls in self.ipt_classes if cls in np.unique(gt_array)]
            for cls in present_classes:
                self.active_samples.append({'image': image_path, 'label': label_path, 'class': cls})

    def __len__(self):
        # 返回活跃样本的长度
        return len(self.active_samples)

    def __getitem__(self, idx):
        # 获取数据
        sample = self.active_samples[idx]
        ct_path, gt_path, cls = sample['image'], sample['label'], sample['class']
        ct_array = self.load_nifti_image(ct_path)  # 加载CT图像
        gt_array = self.load_nifti_image(gt_path)  # 加载标签

        # 生成二值掩码：1代表当前类别，0代表其他类别
        binary_mask = (gt_array == cls).astype(np.int32)
        
        # 构建原始数据字典
        item_ori = {
            'image': ct_array,
            'label': binary_mask,
            'class': cls
        }
        
        # 如果有数据转换，应用转换，否则返回原始数据
        if self.transform is not None:
            item = self.transform(item_ori)
        else:
            item = item_ori
        
        # 标准化键值
        post_item = self.std_keys(item)
        return post_item
    
    def load_nifti_image(self, file_path):
        # 加载NIFTI格式的图像数据
        nifti_image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(nifti_image)
        return image_array
    
    def std_keys(self, post_item):
        # 只保留关键信息键，去掉其他无关键
        keys_to_remain = ['image', 'label', 'class']
        keys_to_remove = post_item.keys() - keys_to_remain
        for key in keys_to_remove:
            del post_item[key]
        return post_item
        
    def process_case_with_clip(gt_voxel_ndarray, ct_shape, categories, text_encoder):
        # 处理数据与CLIP模型的文本嵌入
        gt_masks = []
        organ_names = []
        for cls in categories:
            # 为每个类别生成相应的掩码
            if cls not in np.unique(gt_voxel_ndarray):
                gt_voxel_ndarray_category = np.zeros(ct_shape)
            else:
                gt_voxel_ndarray_category = (gt_voxel_ndarray == cls).astype(np.int32)
            organ_name = "organ_name_for_class_{}".format(cls)  # 为类别生成名称
            organ_names.append(organ_name)
            gt_masks.append(gt_voxel_ndarray_category)
        text_embeddings = text_encoder(organ_names)  # 使用TextEncoder生成文本嵌入
        return np.stack(gt_masks, axis=0), text_embeddings  # 返回类别掩码和文本嵌入

# 自定义的BatchedDistributedSampler，用于分布式训练
class BatchedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, shuffle, batch_size, num_replicas=None, rank=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))  # 计算每个进程应分配的样本数
        self.total_size = self.num_samples * self.num_replicas  # 计算总样本数
        self.batch_size = batch_size  # 每批次的样本数

    def __iter__(self):
        print('run BatchedDistributedSampler iter')
        indices = list(range(len(self.dataset)))  # 获取数据集索引列表

        # 按照每个数据集的长度划分索引
        indices = [indices[i:i + l] for i, l in zip(self.dataset.offsets[:-1], self.dataset.lengths)]

        # 如果需要随机打乱，则打乱每个数据集的索引
        if self.shuffle:
            for idx, subset_indices in enumerate(indices):
                random.shuffle(indices[idx])


        # 处理每个数据集的最后一个批次，确保批次大小一致
        for idx, subset_indices in enumerate(indices):
            r = len(subset_indices) % self.batch_size
            if r > 0:
                indices[idx] = indices[idx][:-r]
        indices = list(itertools.chain(*indices))  # 将所有子数据集的索引合并成一个
        indices = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]  # 按批次划分索引
        if self.shuffle:
            random.shuffle(indices)  # 打乱所有批次顺序
        
        batch_num = len(indices)
        replicas_size = batch_num // self.num_replicas
        start = self.rank * replicas_size  # 当前进程负责的数据起始索引
        end = start + replicas_size if self.rank != self.num_replicas - 1 else batch_num  # 当前进程负责的数据结束索引
        batched_indices = list(itertools.chain(*(indices[start:end])))  # 获取当前进程的所有数据索引
        
        self.total_size = len(indices)  # 更新总样本数
        self.num_samples = self.total_size // self.num_replicas  # 更新每个进程的样本数
        
        return iter(batched_indices)  # 返回当前进程的数据索引迭代器

# 自定义的collate函数，用于批量处理数据
def collate_fn(batch):
    images = []
    labels = []
    classes = []
    
    # 将每个样本的图像、标签和类别分别存入对应的列表
    for sample in batch:
        images.append(sample['image'])
        labels.append(sample['label'])
        classes.append(sample['class'])
    
    # 返回一个批次的数据，包括图像、标签和类别
    return {
        'image': torch.stack(images, dim=0),
        'label': torch.stack(labels, dim=0),
        'class': classes  # 可以用于后续的损失计算和缓冲区更新
    }

# 最小-最大归一化转换：将图像的像素值归一化到[0, 1]之间
class MinMaxNormalization(transforms.Transform):
    def __call__(self, data):
        d = dict(data)
        k = "image"
        d[k] = d[k] - d[k].min()  # 减去最小值
        d[k] = d[k] / np.clip(d[k].max(), a_min=1e-8, a_max=None)  # 除以最大值，防止除零
        return d

# 维度转换：交换图像的某些维度
class DimTranspose(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys  # 需要转换维度的键
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.swapaxes(d[key], -1, -3)  # 交换最后一维和倒数第三维
        return d

# 构建组合数据集：从多个数据集生成一个联合数据集
def build_concat_dataset(root_path, dataset_codes, transform, buffer_manager, clip_ckpt, organ_list):
    concat_dataset = []
    CombinationDataset_len = 0

    for dataset_code in dataset_codes:

        datalist = buffer_manager.get_active_classes()  # 加载每个数据集的JSON文件

        universal_ds = UniversalDataset(
            data=datalist, 
            transform=transform, 
            test_mode=False, 
            buffer_manager=buffer_manager, 
            clip_ckpt=clip_ckpt,
            organ_list=organ_list
        )
        concat_dataset.append(universal_ds)
        CombinationDataset_len += len(universal_ds)  # 计算总数据集长度

    print(f'Dataset loaded, dataset size: {CombinationDataset_len}')
    return UnionDataset(ConcatDataset(concat_dataset), concat_dataset)  # 返回联合数据集

# 获取数据加载器：用于创建训练数据的加载器
def get_loader(args, buffer_manager):
    # 定义一系列数据增强和预处理操作
    train_transform = transforms.Compose(
        [
            transforms.AddChanneld(keys=["image"]),
            DimTranspose(keys=["image", "label"]),
            MinMaxNormalization(),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=args.spatial_size, mode='constant'),
            transforms.OneOf([
                transforms.Resized(keys=["image", "label"], spatial_size=args.spatial_size),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=args.spatial_size,
                    pos=2,
                    neg=1,
                    num_samples=1,
                    image_key="image",
                    image_threshold=0,
                ),
            ], weights=[1, 1]),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.Resized(keys=["image", "label"], spatial_size=args.spatial_size),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    # 打印提示信息，表明正在组合数据集
    print(f'----- train on combination dataset -----')
    # 调用build_concat_dataset构建组合数据集
    combination_train_ds = build_concat_dataset(
        root_path=args.data_dir, 
        dataset_codes=args.dataset_codes, 
        transform=train_transform,
        buffer_manager=buffer_manager,
        clip_ckpt=args.clip_ckpt,
        organ_list=args.organ_list
    )
    
    # 如果使用分布式训练，创建BatchedDistributedSampler采样器
    train_sampler = BatchedDistributedSampler(combination_train_ds, shuffle=True, batch_size=args.batch_size) if args.dist else None
   
    # 创建DataLoader，用于批量加载数据
    train_loader = data.DataLoader(
        combination_train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # 如果没有采样器则打乱数据
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,  # 使用自定义的collate函数
    )
    return train_loader

# 文本与图像特征的拼接函数
def concatenation(args, text_features, image_features, input_ids):
    # 根据input_ids获取图像token的位置掩码
    special_image_token_mask = input_ids == args.image_token_index
    batch_size, text_seq_length = input_ids.shape
    _, image_seq_length, image_hidden_size = image_features.shape
    # 计算最大嵌入维度，包含图像和文本的嵌入
    max_embed_dim = (special_image_token_mask.sum(dim=-1).max() * image_seq_length) + text_seq_length

    # 初始化最终嵌入的tensor
    final_embedding = torch.zeros(batch_size, max_embed_dim, image_hidden_size, dtype=image_features.dtype, device=image_features.device)
    
    # 计算文本和图像特征在最终序列中的位置
    new_token_positions = torch.cumsum(special_image_token_mask * image_seq_length + 1, dim=-1) - 1
    
    # 将文本特征插入到最终的嵌入矩阵中
    non_image_indices = ~special_image_token_mask
    final_embedding.scatter_(1, new_token_positions.unsqueeze(-1).expand(-1, -1, text_features.size(-1)), text_features)
    
    # 将图像特征插入到最终的嵌入矩阵中
    image_positions = (special_image_token_mask.cumsum(dim=-1) * image_seq_length) - image_seq_length
    final_embedding.scatter_(1, image_positions.unsqueeze(-1).expand(-1, -1, image_features.size(-1)), image_features)
    
    return final_embedding  # 返回拼接后的特征
