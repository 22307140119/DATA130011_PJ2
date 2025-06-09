
# 在 CIFAR10 数据集上训练 CNN

### 代码架构
`config.py`: 调整超参配置，含随机种子、损失函数 <br>
`data.py`: 读取 CIFAR10 数据集 <br>
`models.py`: 定义模型结构，含激活函数 <br>
`train.py`: 训练一个 epoch 的训练流程 <br>
`test.py`: 测试模型的流程 <br>
`main.py`: 定义优化器、学习率调度器、组织训练流程、保存模型 <br>


### 训练
训练前检查 `models.py` 模型结构、 `Config.py` 超参配置、 `main.py` 训练流程。
```shell
cd CNN
python main.py
```
最佳模型会自动输出到 config.save_dir/best_model.pth。<br>
训练完成后会自动在测试集上评估模型。


### 测试
测试前检查 `models.py` 模型结构（需要与模型训练时一致）、 `Config.py` 超参配置、 `test.py` 中的 `model_path` 变量。
```shell
python test.py
```

