import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = 0     # modified
num_workers = 0
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.path.abspath(__file__))
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")   # modified
print(device)
print(torch.cuda.get_device_name(0))    # modified



# Initialize your data loader and make sure that dataloader works as expected by observing one sample from it.
# 初始化 dataloader ，并且输出一张照片到 reports\figures\sample_image.png 验证一下
train_loader = get_cifar_loader(batch_size=batch_size, train=True, num_workers=num_workers)
val_loader = get_cifar_loader(batch_size=batch_size, train=False, num_workers=num_workers)
for X,y in train_loader:
    ## --------------------
    # Add code as needed
    print(f"Batch shape: X.shape = {X.shape}, y.shape - {y.shape}")     # 应为 [128, 3, 32, 32] 和 [128]
    plt.imshow(X[0].permute(1, 2, 0).numpy() * 0.5 + 0.5)               # 作用 loaders.py 中的逆变换
    plt.savefig(os.path.join(figures_path, 'sample_image.png'))
    print(f'data loaded, a sample image saved at {figures_path}/sample_image.png')
    ## --------------------
    break



# This function is used to calculate the accuracy of model classification
# 计算模型分类的准确率
def get_accuracy(model, data_loader, device):
    ## --------------------
    # Add code as needed
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total
    ## --------------------


# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire training process. 
# In order to plot the loss landscape, you need to record the loss value of each step.
# Of course, as before, you can test your model after drawing a training round and save the curve to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None, patience=5):
    model.to(device)
    # learning_curve = [np.nan] * epochs_n
    learning_curve = [np.nan]       # 为了让可视化曲线从 epoch 1.0 开始
    train_accuracy_curve = [np.nan]
    val_accuracy_curve = [np.nan]
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0
    count = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        epoch_loss_list = []  # use this to record the loss value of each step 单个 epoch 内的损失
        epoch_grad = []       # use this to record the loss gradient of each step 单个 epoch 内的梯度
        # learning_curve[epoch] = 0  # maintain this to plot the training curve

        for batch_idx, data in enumerate(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)

            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            ## --------------------
            # Add your code
            loss.backward()                     # 反向传播

            epoch_loss_list.append(loss.item()) # 记录损失
            grad = model.classifier[4].weight.grad.clone()  # torch.Size([10, 512])，最后一层的梯度
            epoch_grad.append(torch.norm(grad).item())      # 取二范数，存入列表
            
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:      # 输出训练信息
                print(f'\t Epoch [{epoch+1}/{epochs_n}],\t Batch [{batch_idx+1}/{batches_n}],\t Loss: {loss.item():.4f}')
            ## --------------------


        epoch_avg_loss = sum(epoch_loss_list) / len(epoch_loss_list)    # epoch平均损失
        learning_curve.append(epoch_avg_loss)
        losses_list.extend(epoch_loss_list)     # 损失，直接 extend, 还是一维列表
        grads.extend(epoch_grad)                # 梯度
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))
        axes[0].plot(range(len(learning_curve)), learning_curve)

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        ## --------------------
        # Add code as needed
        model.eval()    # 评估
        correct = 0
        total = 0
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                val_loss = criterion(outputs, y)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                val_losses.append(val_loss.item())
        
        val_accuracy = correct / total
        train_accuracy = get_accuracy(model, train_loader, device)

        # 保存最佳模型
        if val_accuracy > max_val_accuracy:
            count = 0
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch
            torch.save(model.state_dict(), os.path.join(best_model_path, 'best_model.pth'))
            print(f'Best model saved at epoch {epoch+1}, val accuracy: {val_accuracy:.2f}%')
        else:
            count += 1
        
        # 更新准确率曲线
        train_accuracy_curve.append(train_accuracy)
        val_accuracy_curve.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs_n}] | '
              f'Train Loss: {epoch_avg_loss:.4f} | '
              f'Train Acc: {train_accuracy*100:.2f}% | '
              f'Val Acc: {val_accuracy*100:.2f}%')
        
        if count > patience:
            print('Early stopping ...')
            break
        

    # 可视化
    plt.figure(figsize=(15, 3))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(learning_curve, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_curve, label='Training Accuracy')
    plt.plot(val_accuracy_curve, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # 绘图
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f'learning_curve.png'))
    plt.close()

    ## --------------------

    return losses_list, grads



# 正式训练流程
# Train your model
# feel free to modify
epo = 20
loss_save_path = os.path.join(home_path, 'reports', 'texts')
grad_save_path = os.path.join(home_path, 'reports', 'texts')
os.makedirs(loss_save_path, exist_ok=True)
os.makedirs(grad_save_path, exist_ok=True)

set_random_seeds(seed_value=2020, device=device)    # 随机种子

lrlist = [2e-3, 1e-3, 5e-4, 2e-4]

for lr in lrlist:
    model = VGG_A()               # 不带 bn 的 VGG
    # model = VGG_A_BatchNorm()     # 带 bn 的 VGG

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()

    loss, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo, best_model_path=models_path, patience=20)    # 训练
    if isinstance(model, VGG_A):
        suffix = 'VGG'
    if isinstance(model, VGG_A_BatchNorm):
        suffix = 'BN'

    # 以追加方式写入文件，一行对应一个模型的 loss hist 和 grad hist
    loss_file = os.path.join(loss_save_path, f'loss_{suffix}.txt')
    grad_file = os.path.join(grad_save_path, f'grads_{suffix}.txt')
    with open(loss_file, 'a') as f:
        np.savetxt(f, [loss], fmt='%s', delimiter=' ')
    with open(grad_file, 'a') as f:
        np.savetxt(f, [grads], fmt='%s', delimiter=' ')
