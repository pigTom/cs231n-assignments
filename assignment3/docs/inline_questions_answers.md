# Transformer Captioning - Inline Questions 答案

## Inline Question 2: ViT 在小数据集上的性能

### 问题
Despite their recent success in large-scale image recognition tasks, ViTs often lag behind traditional CNNs when trained on smaller datasets. What underlying factor contribute to this performance gap? What techniques can be used to improve the performance of ViTs on small datasets?

### 答案

#### 性能差距的根本原因

**1. 缺乏归纳偏置（Inductive Bias）**

Vision Transformers lack the strong inductive biases inherent in CNNs, such as locality and translation equivariance. CNNs encode these priors directly into their architecture through local receptive fields and weight sharing across spatial locations. Without sufficient training data, ViTs struggle to learn these fundamental image properties from scratch, requiring substantially more examples to discover patterns that CNNs assume by design.

**关键点详解：**

```
CNN 的内置假设:
├─ 局部性（Locality）
│  └─ 卷积核只看局部区域 → 天然学习局部特征
├─ 平移不变性（Translation Equivariance）
│  └─ 权重共享 → 同样的特征检测器用于整个图像
└─ 层次结构（Hierarchical）
   └─ 逐层增大感受野 → 从局部到全局

ViT 的情况:
├─ 全局注意力（Global Attention）
│  └─ 第一层就能看到所有 patch → 需要数据学习何时关注局部
├─ 无权重共享
│  └─ 每个位置独立学习 → 需要更多数据学习平移不变性
└─ 均匀处理
   └─ 所有层都是相同结构 → 需要数据学习层次特征
```

**2. 更高的模型容量与自由度**

```
参数使用效率:
  CNN:  强约束 → 参数少但利用率高
  ViT:  弱约束 → 参数多但需要大量数据来有效利用

小数据集场景:
  CNN:  ✓ 归纳偏置指导学习
  ViT:  ✗ 过拟合，无法泛化
```

#### 改进技术

**1. 数据增强（Data Augmentation）** - 最重要！

```python
# 强数据增强对 ViT 至关重要
transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomRotation(15),
    # 高级增强
    RandAugment(num_ops=2, magnitude=9),
    # Mixup / Cutmix
    ...
])

效果：在小数据集上，强数据增强可以提升 5-10% 准确率
```

**2. 预训练与迁移学习（Pre-training & Transfer Learning）**

```
策略：
1. 在大数据集（ImageNet）上预训练
2. 在目标小数据集上微调

为什么有效：
  预训练学到了通用的视觉特征和归纳偏置
  微调只需要适应特定任务
```

**3. 正则化技术**

```python
# Dropout
model = VisionTransformer(dropout=0.3)  # 小数据集用更大的 dropout

# Weight Decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    weight_decay=1e-3  # 小数据集用更强的正则化
)

# Stochastic Depth
# 随机丢弃层，类似 Dropout 但用于层
```

**4. 知识蒸馏（Knowledge Distillation）**

```
Teacher (大 CNN) → Student (小 ViT)
  用训练好的 CNN 指导 ViT 学习
  DeiT (Data-efficient ViT) 使用这个技术
```

**5. 混合架构（Hybrid Architectures）**

```
CNN Stem + ViT Body:
  使用 CNN 提取低层特征（引入归纳偏置）
  使用 ViT 处理高层语义（全局建模能力）

示例：
  Early Convolutions ViT
  ConViT（卷积与注意力的混合）
```

**6. 减小模型容量**

```python
# 小数据集用小模型
model = VisionTransformer(
    embed_dim=128,      # 更小的维度
    num_layers=4,       # 更少的层
    num_heads=4,
    patch_size=8        # 更大的 patch（减少序列长度）
)
```

### 总结表格

| 技术 | 效果 | 实现难度 | 推荐优先级 |
|------|------|---------|-----------|
| **强数据增强** | +++++ | 简单 | ⭐⭐⭐⭐⭐ |
| **预训练** | +++++ | 中等 | ⭐⭐⭐⭐⭐ |
| **正则化** | ++++ | 简单 | ⭐⭐⭐⭐ |
| **减小模型** | +++ | 简单 | ⭐⭐⭐⭐ |
| **知识蒸馏** | ++++ | 复杂 | ⭐⭐⭐ |
| **混合架构** | +++ | 中等 | ⭐⭐⭐ |

---

## Inline Question 3: ViT Self-Attention 计算成本分析

### 问题
How does the computational cost of the self-attention layers in a ViT change if we independently make the following changes? Please ignore the computation cost of QKV and output projection.

(i) Double the hidden dimension.
(ii) Double the height and width of the input image.
(iii) Double the patch size.
(iv) Double the number of layers.

### 答案

#### 前置知识：Self-Attention 的计算复杂度

Self-attention 的主要计算步骤：

```python
# 1. 计算注意力分数（最耗时）
scores = Q @ K^T  # (N, num_patches, D/H) @ (N, D/H, num_patches)
                  # → (N, num_patches, num_patches)

# 2. Softmax
attn_weights = softmax(scores / √(D/H))  # (N, num_patches, num_patches)

# 3. 加权求和
output = attn_weights @ V  # (N, num_patches, num_patches) @ (N, num_patches, D/H)
                           # → (N, num_patches, D/H)
```

**关键变量：**
- `N` = batch size
- `L` = sequence length (num_patches) = (H/P) × (W/P)
- `D` = hidden dimension
- `H` = number of heads
- `d` = D/H = dimension per head

**计算复杂度分析：**

```
Step 1: Q @ K^T
  每个元素: O(d) 次乘法
  总共: L × L 个元素
  复杂度: O(L² × d)

Step 2: Softmax
  复杂度: O(L²)

Step 3: Attention @ V
  每个元素: O(L) 次乘法
  总共: L × d 个元素
  复杂度: O(L² × d)

总计算复杂度: O(L² × d) = O(L² × D/H)

对于多头注意力（H 个头）:
  总复杂度: O(L² × D)
```

**重要结论：**
```
Self-Attention 的计算成本 ∝ L² × D

其中:
  L = num_patches = (Image_Height / Patch_Size) × (Image_Width / Patch_Size)
  D = hidden_dim
```

---

#### (i) Double the hidden dimension

**分析：**

```
原始: O(L² × D)
加倍维度: O(L² × 2D) = 2 × O(L² × D)
```

**答案：**

**Computational cost increases by a factor of 2 (doubles).**

The attention mechanism computes L² attention weights, and each weight requires D operations for the weighted sum over value vectors. Doubling D directly doubles the computational cost since the sequence length L remains unchanged.

**详细说明：**

```
假设原始配置:
  L = 16 patches (4×4 grid with patch_size=8 for 32×32 image)
  D = 128

计算步骤:
  Attention scores: (16×16) × (128/H) 次乘法
  Weighted sum: (16×16) × (128/H) 次乘法
  总成本: O(16² × 128)

加倍 D 到 256:
  Attention scores: (16×16) × (256/H) 次乘法
  Weighted sum: (16×16) × (256/H) 次乘法
  总成本: O(16² × 256) = 2 × O(16² × 128)
```

---

#### (ii) Double the height and width of the input image

**分析：**

```
原始图像: H × W
新图像: 2H × 2W (面积 × 4)

Patch 数量:
  原始: L = (H/P) × (W/P)
  新的: L' = (2H/P) × (2W/P) = 4L

计算复杂度:
  原始: O(L² × D)
  新的: O((4L)² × D) = O(16L² × D) = 16 × O(L² × D)
```

**答案：**

**Computational cost increases by a factor of 16.**

Doubling both image dimensions quadruples the number of patches (from L to 4L), since patches are extracted from a 2D grid. The self-attention complexity scales quadratically with sequence length, so (4L)² = 16L², resulting in a 16× increase in computational cost.

**详细说明：**

```
示例: 32×32 图像 with patch_size=8

原始:
  Grid: 4×4 = 16 patches
  Cost: O(16² × D) = O(256D)

双倍尺寸: 64×64 图像
  Grid: 8×8 = 64 patches = 4 × 16
  Cost: O(64² × D) = O(4096D) = 16 × O(256D)

关键洞察:
  图像维度 × 2 → patch 数量 × 4 → 计算成本 × 16
  这是 ViT 处理高分辨率图像的主要瓶颈！
```

**实际影响：**

```
分辨率    Patches   计算成本倍数
32×32     16        1×
64×64     64        16×      ← 双倍尺寸
128×128   256       256×     ← 4倍尺寸
224×224   784       2401×    ← ImageNet 标准尺寸

这就是为什么 ViT 在高分辨率图像上很慢！
```

---

#### (iii) Double the patch size

**分析：**

```
Patch size × 2 → 每个维度的 patch 数量 / 2

Patch 数量:
  原始: L = (H/P) × (W/P)
  新的: L' = (H/2P) × (W/2P) = L/4

计算复杂度:
  原始: O(L² × D)
  新的: O((L/4)² × D) = O(L²/16 × D) = (1/16) × O(L² × D)
```

**答案：**

**Computational cost decreases by a factor of 16 (becomes 1/16 of original).**

Doubling the patch size reduces the number of patches by a factor of 4 (since patches are 2D), and since self-attention scales quadratically with sequence length, the computational cost reduces by (1/4)² = 1/16. This is a common strategy to reduce computational cost for high-resolution images.

**详细说明：**

```
示例: 32×32 图像

原始 patch_size=4:
  Grid: 8×8 = 64 patches
  Cost: O(64² × D) = O(4096D)

双倍 patch_size=8:
  Grid: 4×4 = 16 patches = 64/4
  Cost: O(16² × D) = O(256D) = (1/16) × O(4096D)

权衡（Trade-off）:
  优点: 计算成本大幅降低
  缺点: 损失细粒度的空间信息
```

**实际应用：**

```
ViT 变体:
  ViT-B/16: patch_size=16, 适中的计算成本
  ViT-B/32: patch_size=32, 更快但性能略差
  ViT-B/8:  patch_size=8,  更慢但性能更好

ImageNet (224×224):
  Patch 16×16 → 196 patches → 可接受
  Patch 8×8   → 784 patches → 很慢（16×）
  Patch 32×32 → 49 patches  → 很快（1/16）
```

---

#### (iv) Double the number of layers

**分析：**

```
每层的计算成本: O(L² × D)
层数 × 2 → 总成本 × 2
```

**答案：**

**Computational cost increases by a factor of 2 (doubles).**

Each transformer layer performs self-attention independently. Doubling the number of layers simply means performing the same O(L² × D) computation twice as many times, resulting in a linear (2×) increase in total computational cost.

**详细说明：**

```
原始: 6 layers
  每层: O(L² × D)
  总计: 6 × O(L² × D)

双倍: 12 layers
  每层: O(L² × D)
  总计: 12 × O(L² × D) = 2 × [6 × O(L² × D)]

关键点:
  层数增加是线性的（1×, 2×, 3×, ...）
  而序列长度增加是二次的（1×, 4×, 9×, ...）

因此增加层数比增加图像分辨率便宜得多！
```

---

### 总结对比表

| 改变 | Sequence Length (L) | Hidden Dim (D) | 计算复杂度 | 成本变化 |
|------|--------------------|--------------|-----------|---------| |
| **(i) 维度 × 2** | L | 2D | O(L² × 2D) | **2×** |
| **(ii) 图像尺寸 × 2** | 4L | D | O(16L² × D) | **16×** |
| **(iii) Patch 尺寸 × 2** | L/4 | D | O(L²/16 × D) | **1/16×** |
| **(iv) 层数 × 2** | L | D | 2 × O(L² × D) | **2×** |

### 关键洞察

```
计算成本排序（从最昂贵到最便宜）:
  1. 增大图像尺寸    → 16× ⚠️ 非常昂贵！
  2. 增加维度        → 2×
  3. 增加层数        → 2×
  4. 增大 patch 尺寸 → 1/16× ✓ 很便宜！

实际建议:
  - 需要提升性能: 增加层数或维度（成本适中）
  - 需要处理高分辨率: 增大 patch size（降低成本）
  - 避免: 盲目增加图像分辨率（成本暴涨）
```

### 计算示例

假设基准配置：
- 图像: 32×32
- Patch size: 8
- Hidden dim: 128
- Layers: 6

```
基准成本:
  L = (32/8)² = 16
  Cost_per_layer = O(16² × 128) = O(32,768)
  Total_cost = 6 × 32,768 = 196,608

场景对比:
(i)   D→256:     6 × O(16² × 256)    = 393,216    (2×)
(ii)  64×64:     6 × O(64² × 128)    = 3,145,728  (16×)  ⚠️
(iii) Patch→16:  6 × O(4² × 128)     = 12,288     (1/16×) ✓
(iv)  Layers→12: 12 × O(16² × 128)   = 393,216    (2×)
```

---

## 📚 参考文献

1. **ViT**: Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020)
2. **DeiT**: Touvron et al., "Training data-efficient image transformers" (2021)
3. **Attention Is All You Need**: Vaswani et al. (2017)
4. **Data Augmentation**: Shorten & Khoshgoftaar, "A survey on Image Data Augmentation" (2019)

---

**提示**: 这些答案可以直接填入 notebook 的对应位置！
