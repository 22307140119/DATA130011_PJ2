import matplotlib.pyplot as plt
import os

module_path = os.path.dirname(os.path.abspath(__file__))
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
os.makedirs(figures_path, exist_ok=True)


def read_txt(filename):
    result = []
    with open(filename, 'r') as file:     # 打开文件
        for line in file:                 # 逐行读取
            # 分割字符串并转换为浮点数
            # 跳过空行处理（if line.strip() 可防止空行报错）
            numbers = [float(num) for num in line.split()]  
            result.append(numbers)        # 添加到结果列表
    return result


# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models on the same step, 
# add it to max_curve, and the minimum value to min_curve
def extract_min_max(data):  # data 是二维嵌套列表，一行对应一个模型，一列对应一个 step
    # min_curve = []
    # max_curve = []
    ## --------------------
    # Add your code
    data_t = list(zip(*data))     # 转置
    min_curve = [min(steploss) for steploss in data_t]
    max_curve = [max(steploss) for steploss in data_t]
    return min_curve, max_curve
    ## --------------------


# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(
        file1=os.path.join(home_path, 'reports/texts/loss_BN.txt'), 
        file2=os.path.join(home_path, 'reports/texts/loss_VGG.txt'),
        label1='Standard VGG + BatchNorm', label2='Standard VGG'):
    ## --------------------
    # Add your code

    # 两个模型每个 steo 的最大和最小损失
    data1 = read_txt(file1)
    min1, max1 = extract_min_max(data1)
    data2 = read_txt(file2)
    min2, max2 = extract_min_max(data2)

    plt.figure(figsize=(10, 4))
    steps = range(len(min1))
    
    # 绘制两条曲线
    plt.plot(steps, min1, 'r', lw=0.5, alpha=0.5)
    plt.plot(steps, max1, 'r', lw=0.5, alpha=0.5)
    plt.fill_between(steps, min1, max1, color='red', alpha=0.2, label=label1)
    
    plt.plot(steps, min2, 'b', lw=0.5, alpha=0.5)
    plt.plot(steps, max2, 'b', lw=0.5, alpha=0.5)
    plt.fill_between(steps, min2, max2, color='blue', alpha=0.2, label=label2)
    
    # 设置图表标题和标签
    plt.title('Loss Landscape')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 2.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 输出图表
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'loss_landscape.png'))
    plt.close()
    ## --------------------


plot_loss_landscape()
