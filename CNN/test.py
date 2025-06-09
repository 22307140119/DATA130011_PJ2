import torch.nn as nn
import torch
import os

from data import get_data
from models import SimpleCNN
from config import Config

# 测试，返回损失和准确率
def test(model, device, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct_cnt = 0
    total_cnt = 0
    
    with torch.no_grad():
        for input, label in test_loader:
            input, label = input.to(device), label.to(device)
            output = model(input)

            if criterion is not None:
                loss = criterion(output, label)
                total_loss += loss.item() * input.size(0)

            pred = output.argmax(dim=1)
            correct_cnt += (pred==label).sum().item()
            total_cnt += label.shape[0]

    return total_loss / total_cnt, correct_cnt / total_cnt

if __name__ == '__main__':
    config = Config()

    model_path = os.path.join(config.save_dir, "best_model.pth")
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))

    trainloader, validloader, testloader = get_data(batch_size=config.batch_size, valid_ratio=config.valid_ratio, seed=config.seed)    # 训练/测试数据
    model = model.to(config.device)                            # 模型
    
    loss, acc = test(model, device=config.device, test_loader=testloader, criterion=config.criterion)
    print(f"Test accuracy: {acc:.4f}")

