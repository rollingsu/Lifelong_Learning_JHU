import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设您已经定义了 BinaryDiceLoss 和 BCELoss
# 如果还没有，请确保导入或定义它们
from utils.loss import BinaryDiceLoss, BCELoss

class EWCLoss(nn.Module):
    """
    适用于弹性权重巩固（EWC）方法的损失函数。
    """
    def __init__(self, criterion, ewc_lambda=0.4, fisher_dict=None, params_dict=None):
        """
        初始化 EWC 损失函数。

        参数：
        - criterion: 基础损失函数，例如组合了 Dice 和 BCE 的损失。
        - ewc_lambda: EWC 的正则化系数。
        - fisher_dict: Fisher 信息矩阵的字典。
        - params_dict: 旧任务的模型参数字典。
        """
        super(EWCLoss, self).__init__()
        self.criterion = criterion
        self.ewc_lambda = ewc_lambda
        self.fisher_dict = fisher_dict if fisher_dict is not None else {}
        self.params_dict = params_dict if params_dict is not None else {}

    def forward(self, outputs, targets, model):
        # 计算基础损失
        loss = self.criterion(outputs, targets)
        
        # 如果存在 Fisher 信息和旧参数，则添加 EWC 正则项
        if self.fisher_dict and self.params_dict:
            for name, param in model.named_parameters():
                if name in self.fisher_dict:
                    # 当前参数 θ_i
                    param_current = param
                    # 旧参数 θ_i^*
                    param_old = self.params_dict[name]
                    # Fisher 信息 F_i
                    fisher = self.fisher_dict[name]
                    # 计算 (θ_i - θ_i^*)^2
                    param_diff = param_current - param_old
                    # 计算 EWC 正则项并累加到损失中
                    loss += (self.ewc_lambda / 2) * (fisher * param_diff.pow(2)).sum()
        
        return loss
