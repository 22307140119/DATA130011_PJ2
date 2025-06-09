import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from models import SimpleCNN
from config import Config
from data import get_data


model_path = 'saved_models/my_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_landscape(model, criterion, loader, resolution=15, save_pth='visualization/loss_landscape.png'):
    # 预加载数据到 GPU
    inputs_list, targets_list = [], []
    for inputs, targets in loader:
        inputs_list.append(inputs.to(device))
        targets_list.append(targets.to(device))
    
    # 模型和参数移到 GPU
    model = model.to(device)
    params = torch.cat([p.data.view(-1) for p in model.parameters()]).to(device)
    dir1 = torch.randn_like(params).normal_(0, 0.1).to(device)
    dir2 = torch.randn_like(params).normal_(0, 0.1).to(device)
    
    # 降低默认分辨率
    x = np.linspace(-0.0001, 0.0001, resolution)
    y = np.linspace(-0.0001, 0.0001, resolution)
    losses = np.zeros((resolution, resolution))
    
    # 计算 loss landsacpe
    with torch.no_grad():
        for i, alpha in enumerate(x):
            for j, beta in enumerate(y):
                perturbed = params + alpha*dir1 + beta*dir2
                offset = 0
                for p in model.parameters():
                    length = p.numel()
                    p.copy_(perturbed[offset:offset+length].view_as(p))
                    offset += length
                
                # 使用预加载的数据计算损失
                running_loss = 0
                for inputs, targets in zip(inputs_list, targets_list):
                    outputs = model(inputs)
                    running_loss += criterion(outputs, targets).item()
                losses[j, i] = running_loss / len(inputs_list)
    
    # 恢复原始参数
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            length = p.numel()
            p.copy_(params[offset:offset+length].view_as(p))
            offset += length
    
    # 绘图
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, losses, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.view_init(elev=30, azim=45)  # 调整角度
    plt.savefig(save_pth, dpi=150, bbox_inches='tight')
    print(f'Plot saved at {save_pth}')


if __name__ == '__main__':

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    config = Config()
    trainloader, _, _ = get_data(batch_size=config.batch_size, valid_ratio=config.valid_ratio, seed=config.seed)
    
    loss_landscape(model, config.criterion, trainloader, resolution=15)  # 测试时先用 10*10 , 最后记得改回 15*15
