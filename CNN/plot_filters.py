import matplotlib.pyplot as plt
import torch
import numpy as np
from models import SimpleCNN


model_path = 'saved_models/my_model.pth'

# 可视化，一次可视化 64 个 filter 的第一个 channel
def plot_filters1(weights, save_pth='visualization/filters.png'):
    plt.figure(figsize=(8, 8))
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.imshow(weights[i][0].cpu().detach(), cmap='viridis')  # 第一通道
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_pth)
    print(f'Plot saved at {save_pth}')


# 可视化，一次可视化 64 个 filter 的三个 channel 叠加
def plot_filters3(weights, save_pth='visualization/filters.png'):
    plt.figure(figsize=(8, 8))
    
    # 确保输入是3通道的卷积核
    assert weights.shape[1] == 3, f"{weights.shape[1]}"
    
    for i in range(min(64, weights.shape[0])):  # 最多显示64个filter
        plt.subplot(8, 8, i+1)
        
        # 获取当前filter的三个通道
        filter_r = weights[i, 0].cpu().detach().numpy()  # R通道
        filter_g = weights[i, 1].cpu().detach().numpy()  # G通道
        filter_b = weights[i, 2].cpu().detach().numpy()  # B通道

        rgb_filter = np.stack([filter_r, filter_g, filter_b], axis=-1)
        # 归一化到[0,1]范围
        rgb_normalized = (rgb_filter - rgb_filter.min()) / (rgb_filter.max() - rgb_filter.min())
        plt.imshow(rgb_normalized)
        plt.axis('off')
        plt.title(f'F{i}', fontsize=6, pad=2)  # 添加filter编号
    
    plt.tight_layout()
    plt.savefig(save_pth, dpi=200, bbox_inches='tight')
    print(f'Filters saved to {save_pth}')



if __name__ == '__main__':
    model = SimpleCNN()
    # 加载模型
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # 获取权重
    # conv1_weights = model.conv1.weight.data
    # plot_filters3(conv1_weights)
    conv2_weights = model.conv2.weight.data
    plot_filters1(conv2_weights)
