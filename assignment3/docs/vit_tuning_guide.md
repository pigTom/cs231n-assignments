# Vision Transformer (ViT) 调优完全指南

## 📋 目录

1. [快速开始](#快速开始)
2. [超参数调优](#超参数调优)
3. [模型架构选择](#模型架构选择)
4. [训练技巧](#训练技巧)
5. [正则化策略](#正则化策略)
6. [诊断和调试](#诊断和调试)
7. [常见问题解决](#常见问题解决)
8. [完整训练配置示例](#完整训练配置示例)

---

## 🚀 快速开始

### 第一步：过拟合测试（必做！）

在开始任何调优之前，**必须**先确保模型能在一个小 batch 上过拟合到 100% 准确率。

```python
import torch
import torch.nn as nn
from cs231n.classifiers.transformer import VisionTransformer

# 1. 创建小数据集
N = 32  # 小batch
X = torch.randn(N, 3, 32, 32)
y = torch.randint(0, 10, (N,))

# 2. 创建模型（关闭 dropout）
model = VisionTransformer(
    img_size=32,
    patch_size=8,
    in_channels=3,
    embed_dim=128,
    num_layers=6,
    num_heads=4,
    dim_feedforward=256,
    num_classes=10,
    dropout=0.0  # ← 关键：过拟合时不要 dropout
)

# 3. 训练配置
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-3,           # ← 较大学习率
    weight_decay=0.0   # ← 无正则化
)
criterion = nn.CrossEntropyLoss()

# 4. 训练循环
for step in range(200):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        acc = (output.argmax(1) == y).float().mean()
        print(f"[{step}/200] Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")

# 期望结果：Accuracy 应该达到 1.0 (100%)
```

**✓ 如果能过拟合** → 实现正确，继续调优
**✗ 如果不能过拟合** → 实现有 bug，先修复再继续

---

## ⚙️ 超参数调优

### 1. 学习率（最重要！）

学习率是**最重要**的超参数，对训练影响最大。

#### 推荐范围

| 数据集大小 | 初始学习率 | 说明 |
|-----------|-----------|------|
| 小数据集 (<10K) | 1e-4 ~ 5e-4 | 较小，防止过拟合 |
| 中等数据集 (10K~100K) | 3e-4 ~ 1e-3 | 中等 |
| 大数据集 (>100K) | 1e-3 ~ 5e-3 | 较大，加速训练 |
| **过拟合测试** | 5e-3 ~ 1e-2 | 最大，快速验证 |

#### 学习率调度器（Scheduler）

```python
# 方案 1: Cosine Annealing（推荐）
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,  # 周期长度
    eta_min=1e-6       # 最小学习率
)

# 使用
for epoch in range(num_epochs):
    train_one_epoch(...)
    scheduler.step()  # 每个 epoch 后更新

# 方案 2: Step Decay
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(
    optimizer,
    step_size=30,   # 每 30 epochs
    gamma=0.1       # 学习率衰减为原来的 0.1
)

# 方案 3: Reduce on Plateau（根据验证集）
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',          # 监控准确率（max）或loss（min）
    factor=0.5,          # 衰减因子
    patience=5,          # 容忍 5 个 epoch 不提升
    verbose=True
)

# 使用
for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_acc = validate(...)
    scheduler.step(val_acc)  # 传入监控指标
```

#### 学习率 Warmup（大模型必需）

```python
class WarmupScheduler:
    """学习率预热 + Cosine Annealing"""
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 base_lr, warmup_lr=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增长
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine Annealing 阶段
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr

# 使用
scheduler = WarmupScheduler(
    optimizer,
    warmup_epochs=5,      # 前 5 个 epoch 预热
    total_epochs=100,
    base_lr=1e-3,
    warmup_lr=1e-6
)

for epoch in range(100):
    lr = scheduler.step()
    print(f"Epoch {epoch}: LR = {lr:.6f}")
    train_one_epoch(...)
```

### 2. Batch Size

#### 推荐值

| GPU 内存 | Batch Size | 说明 |
|---------|-----------|------|
| 4GB | 32-64 | 小模型 |
| 8GB | 64-128 | 中等 |
| 16GB+ | 128-256 | 大模型 |

#### Batch Size 与学习率的关系

**重要规则**：Batch size 增大时，学习率也应该相应增大。

```python
# 线性缩放规则（Linear Scaling Rule）
base_batch_size = 64
base_lr = 1e-3

your_batch_size = 256
your_lr = base_lr * (your_batch_size / base_batch_size)
# your_lr = 1e-3 * (256 / 64) = 4e-3
```

#### 小技巧：梯度累积（Gradient Accumulation）

如果 GPU 内存不够，可以用梯度累积模拟大 batch size：

```python
accumulation_steps = 4  # 累积 4 个 batch
optimizer.zero_grad()

for i, (x, y) in enumerate(dataloader):
    output = model(x)
    loss = criterion(output, y) / accumulation_steps  # 缩放 loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 等效于 batch_size * accumulation_steps 的大 batch
```

### 3. Weight Decay（权重衰减）

Weight decay 是 L2 正则化，防止过拟合。

#### 推荐范围

| 数据集大小 | Weight Decay | 说明 |
|-----------|-------------|------|
| 过拟合测试 | 0.0 | 不要正则化 |
| 小数据集 | 1e-3 ~ 5e-3 | 强正则化 |
| 中等数据集 | 1e-4 ~ 1e-3 | 中等 |
| 大数据集 | 1e-5 ~ 1e-4 | 轻微 |

#### 高级技巧：不对所有参数应用 weight decay

```python
# 不对 bias 和 LayerNorm 参数应用 weight decay
def get_parameter_groups(model, weight_decay):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Bias 和 LayerNorm 参数不应用 weight decay
        if 'bias' in name or 'norm' in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]

# 使用
param_groups = get_parameter_groups(model, weight_decay=1e-4)
optimizer = torch.optim.AdamW(param_groups, lr=1e-3)
```

### 4. Dropout

Dropout 在每个子层之后应用，防止过拟合。

#### 推荐值

| 场景 | Dropout Rate | 说明 |
|------|-------------|------|
| 过拟合测试 | 0.0 | 不要 dropout |
| 小数据集 | 0.3 ~ 0.5 | 强 dropout |
| 中等数据集 | 0.1 ~ 0.3 | 中等 |
| 大数据集 | 0.0 ~ 0.1 | 轻微或无 |

```python
model = VisionTransformer(
    ...,
    dropout=0.1  # 10% dropout
)
```

### 5. Optimizer 选择

#### Adam vs AdamW vs SGD

| Optimizer | 优点 | 缺点 | 推荐场景 |
|-----------|------|------|---------|
| **Adam** | 收敛快，自适应学习率 | 泛化略差 | 快速实验 |
| **AdamW** | 收敛快，更好的正则化 | - | **推荐** |
| **SGD+Momentum** | 泛化最好 | 需要仔细调参 | 最终模型 |

```python
# 推荐：AdamW
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),    # 默认值
    weight_decay=1e-4
)

# 替代：SGD with Momentum
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2,               # SGD 需要更大的学习率
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True          # Nesterov 加速
)
```

---

## 🏗️ 模型架构选择

### 1. Embed Dimension（嵌入维度）

控制模型的"宽度"。

| 模型大小 | Embed Dim | Parameters | 用途 |
|---------|-----------|-----------|------|
| Tiny | 64-128 | ~100K | 快速实验、小数据集 |
| Small | 256-384 | ~1M | 中等数据集 |
| Base | 512-768 | ~10M | 标准配置 |
| Large | 1024 | ~100M | 大数据集、SOTA |

### 2. Number of Heads（注意力头数）

必须满足：`embed_dim % num_heads == 0`

| Embed Dim | 推荐 Heads | Head Dim |
|-----------|-----------|----------|
| 128 | 4 or 8 | 32 or 16 |
| 256 | 4 or 8 | 64 or 32 |
| 512 | 8 | 64 |
| 768 | 12 | 64 |
| 1024 | 16 | 64 |

**经验法则**：Head dimension 保持在 32-64 之间效果最好。

### 3. Number of Layers（层数）

控制模型的"深度"。

| 层数 | 用途 | 说明 |
|------|------|------|
| 2-4 | 快速实验 | 训练快，适合调参 |
| 6-8 | 标准配置 | 性能与速度平衡 |
| 12 | Base 模型 | 需要更多数据 |
| 24+ | Large 模型 | 需要大数据集和长时间训练 |

### 4. Feedforward Dimension

通常是 `embed_dim` 的 2-4 倍。

```python
dim_feedforward = embed_dim * 4  # 标准配置
```

| Embed Dim | Feedforward Dim |
|-----------|----------------|
| 128 | 512 |
| 256 | 1024 |
| 512 | 2048 |
| 768 | 3072 |

### 5. Patch Size（ViT 特有）

Patch size 影响序列长度和计算量。

#### 对于 32×32 图像（CIFAR-10）

| Patch Size | Num Patches | 计算量 | 性能 |
|-----------|-------------|-------|------|
| 4×4 | 64 | 高 | 最好（细粒度）|
| 8×8 | 16 | 中 | 良好（推荐）|
| 16×16 | 4 | 低 | 较差（太粗糙）|

#### 对于 224×224 图像（ImageNet）

| Patch Size | Num Patches | 计算量 | 常用模型 |
|-----------|-------------|-------|---------|
| 16×16 | 196 | 高 | ViT-Base/16 |
| 32×32 | 49 | 低 | ViT-Base/32 |

**经验法则**：
- 图像越大，patch size 可以越大
- Patch size 越小，性能越好，但计算量越大

### 6. 模型配置示例

```python
# Tiny (快速实验)
model = VisionTransformer(
    img_size=32,
    patch_size=8,
    embed_dim=128,
    num_layers=4,
    num_heads=4,
    dim_feedforward=512,
    dropout=0.1
)
# Parameters: ~200K

# Small (中等性能)
model = VisionTransformer(
    img_size=32,
    patch_size=8,
    embed_dim=256,
    num_layers=6,
    num_heads=8,
    dim_feedforward=1024,
    dropout=0.1
)
# Parameters: ~2M

# Base (标准配置)
model = VisionTransformer(
    img_size=32,
    patch_size=8,
    embed_dim=512,
    num_layers=6,
    num_heads=8,
    dim_feedforward=2048,
    dropout=0.1
)
# Parameters: ~10M
```

---

## 🎓 训练技巧

### 1. 数据增强（Data Augmentation）

ViT 比 CNN 更依赖数据增强！

```python
import torchvision.transforms as transforms

# 基础增强（推荐）
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # 随机裁剪
    transforms.RandomHorizontalFlip(p=0.5),    # 随机水平翻转
    transforms.ColorJitter(                    # 颜色抖动
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], # 标准化
                        std=[0.5, 0.5, 0.5])
])

# 强增强（大数据集）
from torchvision.transforms import RandAugment

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=9),      # RandAugment
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
])

# 验证集（只标准化）
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
])
```

### 2. Mixup / Cutmix

高级数据增强技术，在训练时混合样本。

```python
def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# 使用
for x, y in dataloader:
    x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
    output = model(x)
    loss = lam * criterion(output, y_a) + (1 - lam) * criterion(output, y_b)
    loss.backward()
    optimizer.step()
```

### 3. Label Smoothing

减少过拟合，提高泛化。

```python
# PyTorch 内置支持
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 手动实现
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        n_classes = pred.size(-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), confidence)

        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1))

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
```

### 4. 梯度裁剪（Gradient Clipping）

防止梯度爆炸。

```python
# 方法 1: Clip by norm（推荐）
max_grad_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

# 方法 2: Clip by value
max_grad_value = 0.5
torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_value)

# 完整训练循环
for x, y in dataloader:
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
```

### 5. 混合精度训练（Mixed Precision）

加速训练，减少内存使用。

```python
from torch.cuda.amp import autocast, GradScaler

# 创建 GradScaler
scaler = GradScaler()

for x, y in dataloader:
    optimizer.zero_grad()

    # 在 autocast 上下文中进行前向传播
    with autocast():
        output = model(x)
        loss = criterion(output, y)

    # 缩放 loss，反向传播
    scaler.scale(loss).backward()

    # 梯度裁剪（可选）
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 更新权重
    scaler.step(optimizer)
    scaler.update()

# 加速：1.5-2x，内存减少：~50%
```

### 6. Early Stopping

防止过拟合，节省时间。

```python
class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.inf

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        print(f'Validation accuracy increased ({self.val_acc_max:.4f} → {val_acc:.4f}). Saving model...')
        torch.save(model.state_dict(), 'best_model.pt')
        self.val_acc_max = val_acc

# 使用
early_stopping = EarlyStopping(patience=10)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_acc = validate(...)

    early_stopping(val_acc, model)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break
```

---

## 🛡️ 正则化策略

### 1. Stochastic Depth（随机深度）

训练时随机跳过某些层，测试时使用所有层。

```python
class StochasticDepth(nn.Module):
    """随机深度正则化"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, residual):
        if not self.training or self.drop_prob == 0:
            return x + residual

        # 以一定概率跳过当前层
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(
            (x.size(0), 1, 1), dtype=x.dtype, device=x.device
        )
        binary_tensor = torch.floor(random_tensor)

        # 训练时缩放
        return x + residual * binary_tensor / keep_prob

# 在 TransformerEncoderLayer 中使用
class TransformerEncoderLayerWithSD(nn.Module):
    def __init__(self, ..., drop_path=0.1):
        super().__init__()
        self.stochastic_depth = StochasticDepth(drop_path)
        ...

    def forward(self, src):
        # Self-attention
        shortcut = src
        src = self.self_attn(...)
        src = self.dropout(src)
        src = self.stochastic_depth(shortcut, src)  # 使用随机深度
        src = self.norm(src)
        ...
```

### 2. 正则化组合策略

| 数据集大小 | Dropout | Weight Decay | Stochastic Depth | 数据增强 |
|-----------|---------|--------------|------------------|---------|
| 小 (<10K) | 0.3-0.5 | 1e-3 ~ 5e-3 | 0.2-0.3 | 强 |
| 中 (10K~100K) | 0.1-0.3 | 1e-4 ~ 1e-3 | 0.1-0.2 | 中等 |
| 大 (>100K) | 0.0-0.1 | 1e-5 ~ 1e-4 | 0.0-0.1 | 基础 |

---

## 🔍 诊断和调试

### 1. 学习曲线分析

```python
import matplotlib.pyplot as plt

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss 曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curves')

    # Accuracy 曲线
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy Curves')

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()

# 诊断指南
"""
情况 1: Train loss 下降，Val loss 上升
  → 过拟合！
  → 解决：增加 dropout, weight decay, 数据增强

情况 2: Train loss 和 Val loss 都很高
  → 欠拟合！
  → 解决：增大模型容量，降低正则化，增加训练时间

情况 3: Train loss 很低，Val loss 稍高但稳定
  → 正常！轻微过拟合是可以接受的

情况 4: Loss 震荡，不稳定
  → 学习率太大！
  → 解决：降低学习率，使用梯度裁剪
"""
```

### 2. 梯度监控

```python
def monitor_gradients(model):
    """监控梯度统计"""
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

            # 打印每层的梯度
            if param_norm > 10:  # 警告：梯度过大
                print(f"⚠️  Large gradient in {name}: {param_norm:.4f}")

    total_norm = total_norm ** 0.5
    print(f"Total gradient norm: {total_norm:.4f}")

    # 判断
    if total_norm > 100:
        print("❌ Gradient explosion! Consider gradient clipping.")
    elif total_norm < 1e-6:
        print("❌ Gradient vanishing! Check your model.")

# 使用
for epoch in range(num_epochs):
    for x, y in dataloader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        monitor_gradients(model)  # 监控

        optimizer.step()
```

### 3. Attention Visualization

```python
def visualize_attention(model, image, layer_idx=0, head_idx=0):
    """可视化注意力图"""
    model.eval()
    with torch.no_grad():
        # 前向传播，保存注意力权重
        # 需要修改 MultiHeadAttention 的 forward 返回 attention weights
        patches = model.patch_embed(image.unsqueeze(0))
        patches = model.positional_encoding(patches)

        # 获取特定层的注意力权重
        attn_weights = []
        for i, layer in enumerate(model.transformer.layers):
            # 需要在 self_attn 中返回 attention weights
            patches, attn = layer(patches, return_attention=True)
            attn_weights.append(attn)

        # 选择特定层和头
        attn = attn_weights[layer_idx][0, head_idx]  # (num_patches, num_patches)

        # 可视化
        plt.figure(figsize=(10, 10))
        plt.imshow(attn.cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Map - Layer {layer_idx}, Head {head_idx}')
        plt.show()
```

### 4. 快速诊断检查清单

```python
def diagnostic_check(model, dataloader, device='cuda'):
    """全面诊断检查"""
    model.to(device)
    model.train()

    print("=" * 60)
    print("DIAGNOSTIC CHECK")
    print("=" * 60)

    # 1. 检查前向传播
    print("\n1. Forward Pass Check")
    try:
        x, y = next(iter(dataloader))
        x, y = x.to(device), y.to(device)
        output = model(x)
        print(f"   ✓ Input shape: {x.shape}")
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Output range: [{output.min():.4f}, {output.max():.4f}]")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    # 2. 检查反向传播
    print("\n2. Backward Pass Check")
    try:
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        loss.backward()
        print(f"   ✓ Loss: {loss.item():.4f}")

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in model.parameters())
        print(f"   ✓ Gradients present: {has_grad}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    # 3. 检查参数数量
    print("\n3. Model Size Check")
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {num_params:,}")
    print(f"   Trainable parameters: {num_trainable:,}")
    print(f"   Model size: {num_params * 4 / 1024 / 1024:.2f} MB (fp32)")

    # 4. 检查过拟合能力
    print("\n4. Overfitting Test (100 steps)")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    for step in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if step % 25 == 0 or step == 99:
            acc = (output.argmax(1) == y).float().mean()
            print(f"   [{step:3d}/100] Loss: {loss.item():.4f}, Acc: {acc:.4f}")

    final_acc = (output.argmax(1) == y).float().mean().item()
    if final_acc > 0.95:
        print(f"   ✓ Can overfit! Final accuracy: {final_acc:.4f}")
    else:
        print(f"   ✗ Cannot overfit. Final accuracy: {final_acc:.4f}")
        print(f"   → Check implementation or increase learning rate")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

# 使用
diagnostic_check(model, train_loader)
```

---

## ❓ 常见问题解决

### 问题 1: 训练 loss 不下降

**可能原因**：
- 学习率太小
- 学习率太大（导致震荡）
- 实现有 bug

**解决方案**：
```python
# 1. 先做过拟合测试
# 如果能过拟合 → 学习率问题
# 如果不能过拟合 → 实现bug

# 2. 尝试不同学习率
for lr in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
    print(f"\nTrying lr={lr}")
    test_learning_rate(model, dataloader, lr, num_steps=50)

# 3. 检查梯度
monitor_gradients(model)
```

### 问题 2: 验证集准确率低于训练集很多

**可能原因**：过拟合

**解决方案**：
```python
# 增加正则化
model = VisionTransformer(..., dropout=0.2)  # 增大 dropout

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-3  # 增大 weight decay
)

# 增加数据增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
```

### 问题 3: Loss 变成 NaN

**可能原因**：
- 学习率太大
- 梯度爆炸
- 数值不稳定

**解决方案**：
```python
# 1. 降低学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 更小

# 2. 添加梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. 检查输入数据
print(f"Input range: [{x.min()}, {x.max()}]")
# 应该标准化到 [-1, 1] 或 [0, 1]

# 4. 使用更稳定的初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

model.apply(init_weights)
```

### 问题 4: GPU 内存不足（OOM）

**解决方案**：
```python
# 1. 减小 batch size
batch_size = 32  # 从 128 降到 32

# 2. 使用梯度累积
accumulation_steps = 4

# 3. 使用混合精度
from torch.cuda.amp import autocast
with autocast():
    output = model(x)

# 4. 减小模型尺寸
model = VisionTransformer(
    embed_dim=128,      # 从 512 降到 128
    num_layers=4,       # 从 6 降到 4
    ...
)

# 5. 清理缓存
torch.cuda.empty_cache()
```

### 问题 5: 训练速度太慢

**解决方案**：
```python
# 1. 使用混合精度训练（1.5-2x 加速）
from torch.cuda.amp import autocast, GradScaler

# 2. 增大 batch size（如果内存允许）
batch_size = 256  # 更大的 batch

# 3. 使用更多 workers
train_loader = DataLoader(dataset, batch_size=128, num_workers=4)

# 4. Pin memory
train_loader = DataLoader(dataset, batch_size=128, pin_memory=True)

# 5. 减小模型尺寸或层数
model = VisionTransformer(..., num_layers=4)  # 从 6 降到 4
```

---

## 📋 完整训练配置示例

### 示例 1: CIFAR-10 小数据集（推荐开始）

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cs231n.classifiers.transformer import VisionTransformer

# ===== 1. 数据准备 =====
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=train_transform)
val_dataset = datasets.CIFAR10(root='./data', train=False,
                               transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=128,
                         shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256,
                       shuffle=False, num_workers=2, pin_memory=True)

# ===== 2. 模型配置 =====
model = VisionTransformer(
    img_size=32,
    patch_size=8,
    in_channels=3,
    embed_dim=256,          # 中等尺寸
    num_layers=6,
    num_heads=8,
    dim_feedforward=1024,
    num_classes=10,
    dropout=0.1
).cuda()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ===== 3. 训练配置 =====
num_epochs = 100
base_lr = 1e-3
weight_decay = 1e-4

# 优化器：AdamW
param_groups = get_parameter_groups(model, weight_decay)
optimizer = torch.optim.AdamW(param_groups, lr=base_lr)

# 学习率调度器：Cosine Annealing with Warmup
scheduler = WarmupScheduler(
    optimizer,
    warmup_epochs=5,
    total_epochs=num_epochs,
    base_lr=base_lr,
    warmup_lr=1e-6
)

# 损失函数：带 Label Smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Early Stopping
early_stopping = EarlyStopping(patience=15)

# 混合精度
scaler = GradScaler()

# ===== 4. 训练循环 =====
best_val_acc = 0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    # 更新学习率
    lr = scheduler.step()

    # ===== 训练 =====
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()

        # 混合精度前向传播
        with autocast():
            output = model(x)
            loss = criterion(output, y)

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新权重
        scaler.step(optimizer)
        scaler.update()

        # 统计
        train_loss += loss.item() * x.size(0)
        pred = output.argmax(dim=1)
        train_correct += (pred == y).sum().item()
        train_total += x.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    # ===== 验证 =====
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda()
            output = model(x)
            loss = criterion(output, y)

            val_loss += loss.item() * x.size(0)
            pred = output.argmax(dim=1)
            val_correct += (pred == y).sum().item()
            val_total += x.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    # 记录
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # 打印
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"LR: {lr:.6f} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_vit_model.pt')
        print(f"✓ Best model saved! Val Acc: {val_acc:.4f}")

    # Early Stopping
    early_stopping(val_acc, model)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

# ===== 5. 绘制学习曲线 =====
plot_learning_curves(train_losses, val_losses, train_accs, val_accs)

print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
```

### 示例 2: 快速实验配置

```python
# 用于快速调参和实验
model = VisionTransformer(
    img_size=32,
    patch_size=8,
    embed_dim=128,          # 小模型
    num_layers=4,           # 少层数
    num_heads=4,
    dim_feedforward=512,
    num_classes=10,
    dropout=0.1
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 50  # 更少 epochs

# 训练速度：~2-3x 快于标准配置
```

---

## 📊 超参数调优流程图

```
开始
  ↓
[1] 过拟合一个 batch
  ├─ ✓ 成功 → 继续
  └─ ✗ 失败 → 修复实现bug，返回步骤1
  ↓
[2] 选择基础配置
  ├─ 小模型：embed_dim=128, layers=4
  ├─ 中模型：embed_dim=256, layers=6
  └─ 大模型：embed_dim=512, layers=12
  ↓
[3] 调整学习率
  ├─ 从 1e-3 开始
  ├─ 观察 loss 曲线
  ├─ 太高：震荡 → 降低
  └─ 太低：收敛慢 → 增大
  ↓
[4] 检查过拟合程度
  ├─ 严重过拟合 → 增加正则化
  │   ├─ 增大 dropout (0.1 → 0.3)
  │   ├─ 增大 weight_decay (1e-4 → 1e-3)
  │   └─ 增加数据增强
  ├─ 欠拟合 → 减少正则化 / 增大模型
  └─ 轻微过拟合 → 正常，微调即可
  ↓
[5] 添加训练技巧
  ├─ 学习率调度器（Cosine Annealing）
  ├─ Warmup（大模型必需）
  ├─ Label Smoothing
  ├─ Mixup/Cutmix（可选）
  └─ 混合精度训练（加速）
  ↓
[6] 最终调优
  ├─ 网格搜索关键超参数
  ├─ 多次运行取平均
  └─ 选择最佳配置
  ↓
完成！
```

---

## 🎯 总结：调优优先级

### 必须做 ✅
1. **过拟合测试** - 验证实现正确
2. **学习率** - 最重要的超参数
3. **数据增强** - ViT 的必需品
4. **基础正则化** - Dropout + Weight Decay

### 应该做 ⭐
5. **学习率调度** - Cosine Annealing
6. **梯度裁剪** - 防止梯度爆炸
7. **Early Stopping** - 节省时间
8. **混合精度** - 加速训练

### 可选做 💡
9. **Warmup** - 大模型有帮助
10. **Label Smoothing** - 提升泛化
11. **Mixup/Cutmix** - 高级增强
12. **Stochastic Depth** - 深层网络

---

**记住**：没有万能的配置，需要根据具体数据集和任务进行调整。从简单配置开始，逐步添加复杂技巧！

📖 相关文档：
- `transformer_residual_pattern.md` - 原理详解
- `transformer_quick_reference.md` - 快速参考
