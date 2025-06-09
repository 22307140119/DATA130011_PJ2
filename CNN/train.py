import torch.nn.functional as F

# 训练一个 epoch，返回训练集损失
def train(model, device, train_loader, optimizer, criterion):
    
    model.train()
    total_loss = 0
    total_samples = 0

    for batch_idx, (input, label) in enumerate(train_loader):
        input, label = input.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(input)
        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * input.size(0)
        total_samples += input.size(0)
    
    return total_loss / total_samples

