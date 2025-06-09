import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 难易样本调节因子

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return loss.mean()



def set_seed(seed):
    random.seed(seed)                   # Python 随机数种子
    np.random.seed(seed)                # NumPy 随机数种子
    torch.manual_seed(seed)             # PyTorch 随机数种子
    torch.cuda.manual_seed(seed)        # 如果使用GPU（CUDA）
    torch.cuda.manual_seed_all(seed)    # 多GPU时
    torch.backends.cudnn.deterministic = True   # 禁止CUDA的随机优化（可能牺牲少量性能）
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子（避免哈希随机化）


class Config:
    seed = 42
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 32
    valid_ratio = 0.2
    # criterion = FocalLoss(alpha=0.25, gamma=2)
    criterion = nn.CrossEntropyLoss()
    epochs = 10

    lr = 2e-3       # SGD(1e-2), SGDMomentum(2e-3), AdamW(4e-5)
    momentum = 0.9
    step_size = 5
    gamma = 0.2

    save_dir = 'saved_models'

