import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os

from data import get_data
from models import SimpleCNN
from train import train
from test import test
from config import Config


def main(config):

    model = SimpleCNN()

    trainloader, validloader, testloader = get_data(batch_size=config.batch_size, valid_ratio=config.valid_ratio, seed=config.seed)    # 训练/测试数据
    model = model.to(config.device)                            # 模型
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)    # 优化器
    # optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)    # 学习率调度器
    

    # 训练
    best_loss = np.Inf
    best_model_path = os.path.join(config.save_dir, "best_model.pth")
    for epoch in range(config.epochs):

        loss = train(model, config.device, trainloader, optimizer, config.criterion)      # 无验证集则按照训练集 loss 保存模型
        print(f'Epoch {epoch+1}/{config.epochs}\tTrain loss: {loss:.4f}', end='')
        if validloader is not None:
            loss, valid_acc = test(model, config.device, validloader, config.criterion)   # 有验证集则按照验证集 loss 保存模型
            print(f'\tValidation loss: {loss:.4f}, Validation accuracy: {valid_acc:.4f}')
        
        if loss < best_loss:    # 保存最小 loss 对应的模型
            best_loss = loss
            os.makedirs(config.save_dir, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f'\tBest model saved at {best_model_path}')

        scheduler.step()


    # 完成训练后测试
    print('Training completed. Evaluating best model on test set ...')
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = test(model, config.device, testloader, config.criterion)
    print(f"Best model test accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    config = Config()
    main(config)
