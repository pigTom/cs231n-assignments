# Transformer 快速参考

## 🎯 核心模式（必背）

```python
# 每个 Transformer 子层的标准模式
shortcut = x           # 1. 保存输入
x = SubLayer(x)        # 2. 应用变换（Attention 或 FFN）
x = Dropout(x)         # 3. 正则化
x = x + shortcut       # 4. 残差连接（梯度高速公路）
x = LayerNorm(x)       # 5. 标准化（稳定训练）
```

## 📊 核心作用

```
┌──────────────┬────────────────────────────────────────┐
│ Residual     │ 允许梯度直接传播，训练深层网络          │
│ (残差连接)   │ ∂Loss/∂x = ... × (∂F/∂x + 1) ← "+1"  │
├──────────────┼────────────────────────────────────────┤
│ LayerNorm    │ 标准化输出分布，稳定训练                │
│ (层标准化)   │ 均值=0, 方差=1                         │
├──────────────┼────────────────────────────────────────┤
│ Dropout      │ 随机丢弃，防止过拟合                    │
│              │ 训练时: 10% 置零，推理时: 无操作        │
└──────────────┴────────────────────────────────────────┘
```

## 🏗️ Transformer 架构

### Encoder Layer (2 个 block)

```
Input (N, S, D)
    ↓
┌─────────────────────┐
│ Self-Attention      │  Q=K=V=src, 双向
│ + Residual + Norm   │
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ Feedforward         │  FC → GELU → FC
│ + Residual + Norm   │
└──────┬──────────────┘
       ↓
Output (N, S, D)
```

### Decoder Layer (3 个 block)

```
Input (N, T, D)
    ↓
┌─────────────────────┐
│ Self-Attention      │  Q=K=V=tgt, 单向 (causal mask)
│ + Residual + Norm   │
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ Cross-Attention     │  Q=tgt, K=V=memory
│ + Residual + Norm   │
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ Feedforward         │  FC → GELU → FC
│ + Residual + Norm   │
└──────┬──────────────┘
       ↓
Output (N, T, D)
```

## 🔑 关键区别

### Self-Attention vs Cross-Attention

```python
# Self-Attention: 序列关注自己
out = self.self_attn(query=tgt, key=tgt, value=tgt, mask=tgt_mask)
#                         ↑        ↑      ↑       ↑
#                         都来自 tgt      有 mask (decoder)

# Cross-Attention: decoder 关注 encoder
out = self.cross_attn(query=tgt, key=memory, value=memory)
#                          ↑          ↑           ↑
#                      来自 tgt    来自 encoder   无 mask
```

### Encoder vs Decoder

| 特性 | Encoder | Decoder |
|------|---------|---------|
| **Block 数量** | 2 (Attn + FFN) | 3 (Self + Cross + FFN) |
| **Attention 类型** | 双向 Self-Attention | 单向 Self + Cross |
| **Mask** | 可选（padding mask） | 必需（causal mask） |
| **用途** | 编码输入序列 | 生成输出序列 |

## 📐 数学公式

### Multi-Head Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) W^O
  where headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)
```

### Layer Normalization

```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β

其中:
  μ = mean(x)      # 沿特征维度
  σ² = var(x)      # 沿特征维度
  γ, β 是可学习参数
```

### Positional Encoding

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

## 🎨 Vision Transformer (ViT)

```
Image (N, C, H, W)
    ↓ Patch Embedding
Patches (N, num_patches, D)
    ↓ + Positional Encoding
    ↓ Transformer Encoder (多层)
Features (N, num_patches, D)
    ↓ Global Average Pooling
    ↓ Classification Head
Logits (N, num_classes)
```

## 🖼️ Image Captioning Transformer

```
┌──────────────┐              ┌──────────────┐
│    Image     │              │   Caption    │
│  (N, D)      │              │   (N, T)     │
└──────┬───────┘              └──────┬───────┘
       │                             │
       ↓ Visual Projection           ↓ Embedding + PE
    Memory                          Target
   (N, 1, W)                      (N, T, W)
       │                             │
       │         ┌───────────────────┘
       │         ↓
       │    Transformer Decoder
       │    ┌─────────────────┐
       │    │ Self-Attention  │ ← causal mask
       │    └─────────────────┘
       │         ↓
       └────►Cross-Attention   ← 关注图像
            └─────────────────┘
                 ↓
            Feedforward
            └─────────────────┘
                 ↓
            Output Layer
                 ↓
            Scores (N, T, V)
```

## ⚙️ 超参数建议

### 训练

```python
# 过拟合测试
optimizer = torch.optim.Adam(model.parameters(),
                             lr=5e-3,           # 较大学习率
                             weight_decay=0.0)  # 无正则化
epochs = 150-200

# 正常训练
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-4 to 1e-3,   # 适中学习率
                             weight_decay=1e-4) # 轻微正则化
use_lr_scheduler = True  # 学习率衰减
```

### 模型配置

```python
# 小模型 (快速实验)
embed_dim = 128
num_heads = 4
num_layers = 4
dim_feedforward = 512

# 中型模型 (论文复现)
embed_dim = 512
num_heads = 8
num_layers = 6
dim_feedforward = 2048

# 大模型 (SOTA)
embed_dim = 768 or 1024
num_heads = 12 or 16
num_layers = 12 or 24
dim_feedforward = 3072 or 4096
```

## 🐛 常见错误

### ❌ 错误 1: Dropout 位置错误
```python
# 错误
x = x + self.attn(x)
x = self.dropout(x)  # ✗ 破坏残差路径

# 正确
x = self.dropout(self.attn(x))
x = x + shortcut     # ✓
```

### ❌ 错误 2: 忘记 causal mask
```python
# Decoder self-attention 必须有 mask
tgt_mask = torch.tril(torch.ones(T, T))  # 下三角
out = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
```

### ❌ 错误 3: 形状不匹配
```python
# 图像特征需要添加序列维度
features: (N, D) → (N, 1, W)  # unsqueeze(1)

# Captions 需要 embedding
captions: (N, T) → (N, T, W)  # embedding
```

### ❌ 错误 4: 学习率太小
```python
# 过拟合时
lr = 1e-3  # ✗ 太小
lr = 5e-3  # ✓ 合适

# 正常训练时
lr = 1e-5  # ✗ 太小
lr = 1e-4  # ✓ 合适
```

## 📏 Shape Cheatsheet

```python
# Multi-Head Attention
Input:  (N, S, D)
Q, K, V: (N, S, D) → split → (N, H, S, D/H)
Scores: (N, H, S, S)
Output: (N, S, D)

# Transformer Encoder
Input:  (N, S, D)
Output: (N, S, D)  # 形状不变

# Transformer Decoder
tgt:    (N, T, D)
memory: (N, S, D)
Output: (N, T, D)  # 形状与 tgt 相同

# Vision Transformer
Input:  (N, C, H, W)
Patches: (N, (H/P)*(W/P), D)
Output: (N, num_classes)

# Image Captioning
features: (N, D) → (N, 1, W)
captions: (N, T) → (N, T, W)
scores:   (N, T, V)
```

## 💡 调试技巧

```python
# 1. 检查形状
print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")

# 2. 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm():.4f}")

# 3. 检查 attention weights
with torch.no_grad():
    attn_weights = torch.softmax(scores, dim=-1)
    print(f"Max attention: {attn_weights.max():.4f}")
    print(f"Min attention: {attn_weights.min():.4f}")

# 4. 过拟合一个 batch
# 如果无法过拟合，说明实现有问题
```

## 🎓 核心要点

1. **残差连接** = 梯度高速公路 = 深层网络的关键
2. **LayerNorm** = 稳定训练 = 更大学习率
3. **Multi-Head** = 多视角 = 更丰富的表示
4. **Position Encoding** = 位置信息 = 序列建模的基础
5. **Cross-Attention** = encoder-decoder 连接 = seq2seq 的核心

---

📖 **详细文档**: `transformer_residual_pattern.md`
🔬 **实现代码**: `cs231n/transformer_layers.py`
🧪 **测试脚本**: `test_vit_overfit.py`
