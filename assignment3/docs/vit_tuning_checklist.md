# Vision Transformer 调优检查清单

## ✅ 开始前的必做检查

- [ ] **过拟合测试**
  ```python
  # 能在 32 个样本上达到 100% 准确率吗？
  # 学习率：5e-3, Dropout: 0.0, Weight Decay: 0.0
  # 训练 150-200 步
  # ✓ 能 → 实现正确，继续
  # ✗ 不能 → 有 bug，先修复
  ```

- [ ] **形状检查**
  ```python
  # 输入：(N, 3, 32, 32)
  # Patches: (N, 16, embed_dim)  # 对于 patch_size=8
  # 输出：(N, num_classes)
  ```

- [ ] **梯度检查**
  ```python
  # 所有参数都有梯度吗？
  # 梯度范数在合理范围内吗？(0.1 ~ 10)
  ```

---

## 🎯 基础配置（从这里开始）

### 模型配置

```python
# 快速实验（推荐开始）
VisionTransformer(
    img_size=32,
    patch_size=8,
    embed_dim=128,        # ← 小模型
    num_layers=4,         # ← 少层数
    num_heads=4,
    dim_feedforward=512,
    dropout=0.1
)
# 训练时间：~10-15 分钟/epoch (CIFAR-10)
```

### 训练配置

```python
# AdamW 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,              # ← 从这个开始
    weight_decay=1e-4     # ← 轻微正则化
)

# 基础数据增强
transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# 训练设置
batch_size = 128      # 如果 OOM，降到 64
num_epochs = 100
```

---

## 🔧 调优流程

### Step 1: 学习率调优（最重要！）

- [ ] **测试不同学习率**
  ```python
  # 尝试：[1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
  # 观察：loss 曲线
  #   - 震荡/NaN → 太大
  #   - 下降太慢 → 太小
  #   - 稳定下降 → ✓
  ```

- [ ] **添加学习率调度器**
  ```python
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer,
      T_max=num_epochs,
      eta_min=1e-6
  )
  ```

### Step 2: 检查过拟合/欠拟合

**检查指标**：
```
Train Acc: 0.95, Val Acc: 0.70  → 严重过拟合
Train Acc: 0.85, Val Acc: 0.75  → 轻微过拟合（正常）
Train Acc: 0.60, Val Acc: 0.58  → 欠拟合
```

#### 如果过拟合：

- [ ] **增加 Dropout**
  ```python
  dropout=0.1 → 0.2 或 0.3
  ```

- [ ] **增加 Weight Decay**
  ```python
  weight_decay=1e-4 → 1e-3
  ```

- [ ] **增强数据增强**
  ```python
  # 添加 ColorJitter
  transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
  ```

- [ ] **减小模型**
  ```python
  embed_dim=256 → 128
  num_layers=6 → 4
  ```

#### 如果欠拟合：

- [ ] **增大模型**
  ```python
  embed_dim=128 → 256 或 512
  num_layers=4 → 6
  ```

- [ ] **降低正则化**
  ```python
  dropout=0.3 → 0.1
  weight_decay=1e-3 → 1e-4
  ```

- [ ] **训练更长时间**
  ```python
  num_epochs=100 → 200
  ```

### Step 3: 添加训练技巧

- [ ] **梯度裁剪**
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```

- [ ] **Label Smoothing**
  ```python
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
  ```

- [ ] **混合精度训练**（加速 1.5-2x）
  ```python
  from torch.cuda.amp import autocast, GradScaler
  scaler = GradScaler()

  with autocast():
      output = model(x)
      loss = criterion(output, y)
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  ```

- [ ] **Early Stopping**
  ```python
  patience = 15  # 15 epochs 不提升就停止
  ```

---

## 📊 超参数速查表

### 学习率

| 场景 | 学习率 |
|------|--------|
| 过拟合测试 | 5e-3 ~ 1e-2 |
| 小数据集 (<10K) | 1e-4 ~ 5e-4 |
| 中等数据集 | 3e-4 ~ 1e-3 |
| 大数据集 (>100K) | 1e-3 ~ 5e-3 |

### Batch Size

| GPU | Batch Size |
|-----|-----------|
| 4GB | 32-64 |
| 8GB | 64-128 |
| 16GB+ | 128-256 |

### Weight Decay

| 数据集 | Weight Decay |
|--------|-------------|
| 过拟合测试 | 0.0 |
| 小数据集 | 1e-3 ~ 5e-3 |
| 中等数据集 | 1e-4 ~ 1e-3 |
| 大数据集 | 1e-5 ~ 1e-4 |

### Dropout

| 数据集 | Dropout |
|--------|---------|
| 过拟合测试 | 0.0 |
| 小数据集 | 0.3 ~ 0.5 |
| 中等数据集 | 0.1 ~ 0.3 |
| 大数据集 | 0.0 ~ 0.1 |

### 模型大小

| 类型 | Embed Dim | Layers | Heads | FFN Dim | 参数量 |
|------|-----------|--------|-------|---------|--------|
| Tiny | 128 | 4 | 4 | 512 | ~200K |
| Small | 256 | 6 | 8 | 1024 | ~2M |
| Base | 512 | 6 | 8 | 2048 | ~10M |

---

## 🐛 常见问题快速诊断

### Loss 不下降

```
✓ 能过拟合一个 batch?
  → No: 实现有 bug
  → Yes: 学习率问题
    - 尝试更大的学习率 (1e-3 → 5e-3)
    - 减少正则化
```

### Loss 变成 NaN

```
原因：学习率太大 或 梯度爆炸
解决：
  1. 降低学习率 (1e-3 → 1e-4)
  2. 添加梯度裁剪 (max_norm=1.0)
  3. 检查输入是否标准化
```

### GPU 内存不足

```
解决方案（按优先级）：
  1. 减小 batch_size (128 → 64 → 32)
  2. 使用混合精度训练 (节省 ~50% 内存)
  3. 减小模型 (embed_dim=256 → 128)
  4. 使用梯度累积
```

### 训练太慢

```
加速方法：
  1. 混合精度训练 (1.5-2x)
  2. 增大 batch_size
  3. 使用 pin_memory=True
  4. 增加 num_workers
  5. 减小模型或层数
```

### 验证集准确率远低于训练集

```
过拟合！
  1. 增大 dropout (0.1 → 0.2)
  2. 增大 weight_decay (1e-4 → 1e-3)
  3. 增加数据增强
  4. 减小模型容量
```

---

## 📈 训练监控指标

### 每个 Epoch 应该记录

- [ ] Train Loss
- [ ] Train Accuracy
- [ ] Val Loss
- [ ] Val Accuracy
- [ ] Learning Rate
- [ ] Gradient Norm（可选）

### 判断标准

```python
# 正常训练
Train Loss: 持续下降
Val Loss: 先下降，后稍微上升（轻微过拟合正常）
Train Acc: 持续上升
Val Acc: 上升后稳定

# 异常情况
Loss 震荡 → 学习率太大
Loss 不变 → 学习率太小 或 实现错误
Val Loss 急剧上升 → 严重过拟合
NaN → 梯度爆炸 或 数值不稳定
```

---

## 🎯 推荐调优顺序

### 第一轮：基础配置

1. [ ] 过拟合测试（验证实现）
2. [ ] 选择小模型（快速迭代）
3. [ ] 调整学习率（最重要）
4. [ ] 基础数据增强

**目标**：达到合理的基线性能（~60-70% val acc）

### 第二轮：减少过拟合

5. [ ] 调整 dropout 和 weight decay
6. [ ] 增加数据增强
7. [ ] 添加学习率调度器

**目标**：缩小 train-val gap

### 第三轮：提升性能

8. [ ] 增大模型（如果有必要）
9. [ ] Label smoothing
10. [ ] 梯度裁剪
11. [ ] Warmup（大模型）

**目标**：达到最优性能

### 第四轮：优化效率

12. [ ] 混合精度训练
13. [ ] 调整 batch size
14. [ ] Early stopping

**目标**：更快的训练速度

---

## 💾 完整训练模板

```python
# ===== 配置 =====
config = {
    # 模型
    'embed_dim': 128,
    'num_layers': 4,
    'num_heads': 4,
    'dropout': 0.1,

    # 训练
    'batch_size': 128,
    'num_epochs': 100,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,

    # 其他
    'label_smoothing': 0.1,
    'warmup_epochs': 5,
}

# ===== 模型 =====
model = VisionTransformer(
    img_size=32,
    patch_size=8,
    in_channels=3,
    embed_dim=config['embed_dim'],
    num_layers=config['num_layers'],
    num_heads=config['num_heads'],
    dim_feedforward=config['embed_dim'] * 4,
    num_classes=10,
    dropout=config['dropout']
).cuda()

# ===== 优化器 =====
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['lr'],
    weight_decay=config['weight_decay']
)

# ===== 学习率调度 =====
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config['num_epochs']
)

# ===== 损失函数 =====
criterion = nn.CrossEntropyLoss(
    label_smoothing=config['label_smoothing']
)

# ===== 训练循环 =====
for epoch in range(config['num_epochs']):
    # 训练
    model.train()
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config['grad_clip']
        )

        optimizer.step()

    # 验证
    model.eval()
    # ... validation code ...

    # 更新学习率
    scheduler.step()
```

---

## 🎓 经验总结

### 一定要做 ✅
1. 过拟合测试（验证实现正确）
2. 调整学习率（影响最大）
3. 数据增强（ViT 必需）
4. 监控训练曲线

### 推荐做 ⭐
5. 学习率调度器
6. 梯度裁剪
7. Label smoothing
8. 混合精度（加速）

### 可选做 💡
9. Warmup（大模型）
10. Mixup/Cutmix
11. Stochastic depth
12. 超参数搜索

### 常见错误 ❌
- 没做过拟合测试就开始训练
- 学习率设置不当
- ViT 不用数据增强
- 过度正则化导致欠拟合
- 没有监控训练过程

---

**记住**：从简单开始，逐步添加复杂技巧。每次只改变一个超参数，观察效果！

📖 详细文档：`vit_tuning_guide.md`
