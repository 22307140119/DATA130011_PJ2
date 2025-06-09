### 文件说明
- `data/`: 加载数据。<br>
- `utils/`: 目前负责初始化模型权重。<br>
- `models/`: 定义模型，包含 VGG_A 和 VGG_A_BatchNorm 定义。<br>
- `train.py`: 组织训练流程，输出最佳模型到 `reports/models/best_model.pth`，输出训练日志到 `reports/texts/` 目录，输出学习曲线到 `reports/figures/learning_curve.png`。<br>
- `VGG_Loss_Landsacpe.py`: 根据 `train.py` 输出的训练日志，画出 VGG_A 和 VGG_A_BatchNorm 训练中的损失范围，图片输出到 `reports/figures/loss_landscape.png`。

### 训练
确认 `train.py` 底部的训练配置，可以通过修改 `models_path`, `loss_save_path`, `grad_save_path` 等，修改模型/日志输出到的路径。
```shell
cd VGG_BATCHNORM
python train.py
```

### 可视化 loss landscape
确认已经运行 `train.py` ，得到 `reports/texts/loss_BN.txt` 和 `reports/texts/loss_VGG.txt` 两个文件。<br>
特别注意 `train.py` 会以追加模式写入 .txt 文件。运行前确保 .txt 文件中仅存储必要数据。<br>
文件中的一“行”对应一个模型的训练日志，两个数值之间用空格分隔。<br>
```shell
python VGG_Loss_Landscape.py
```
可视化结果输出到`reports/figures/loss_landscape.py`
