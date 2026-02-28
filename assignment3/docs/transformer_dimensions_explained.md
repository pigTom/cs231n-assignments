# Transformer 中的维度术语详解

## 🎯 简短答案

在 **Inline Question 3 的上下文中**，**hidden dimension 就是 embedding dimension**。

但在更广泛的 Transformer 文献中，这两个术语有细微差别。让我详细解释。

---

## 📐 维度术语对照表

### Vision Transformer (ViT) 中的维度

```python
model = VisionTransformer(
    img_size=32,
    patch_size=8,
    in_channels=3,
    embed_dim=128,        # ← 这就是 "hidden dimension"
    num_layers=6,
    num_heads=4,
    dim_feedforward=512,  # ← 这是 FFN 的中间维度
    num_classes=10,
    dropout=0.1
)
```

| 术语 | 值 | 说明 | 别名 |
|------|-----|------|------|
| **embed_dim** | 128 | Patch 嵌入后的维度 | hidden_dim, d_model, model_dim |
| **dim_feedforward** | 512 | FFN 中间层维度 | ffn_dim, d_ff |
| **head_dim** | 32 | 每个注意力头的维度 | embed_dim / num_heads |
| **num_patches** | 16 | Patch 数量（序列长度） | sequence_length, L |

---

## 🔍 详细解释

### 1. Embedding Dimension (embed_dim)

```
输入图像: (N, 3, 32, 32)
    ↓ Patch Embedding
Patches: (N, 16, 128)  ← 128 就是 embed_dim
    ↓
    每个 patch 是 128 维向量
```

**作用**：
- Patch 嵌入后的维度
- 整个 Transformer 中数据流动的主要维度
- 在每一层中保持不变

**代码中的体现**：
```python
# Patch Embedding
self.proj = nn.Linear(patch_dim, embed_dim)  # 192 → 128

# 之后所有数据都是 (N, num_patches, embed_dim)
x = self.patch_embed(x)         # (N, 16, 128)
x = self.positional_encoding(x) # (N, 16, 128)
x = self.transformer(x)         # (N, 16, 128)
x = torch.mean(x, dim=1)        # (N, 128)
```

### 2. Hidden Dimension - 两种含义

#### 含义 1: Model Dimension (常见) = embed_dim

在大多数 Transformer 文献和代码中，**hidden dimension 就是指 embed_dim**。

```python
# 原始 Transformer 论文术语
d_model = 512        # "model dimension" = embed_dim
d_ff = 2048          # "feedforward dimension"

# PyTorch 实现
nn.TransformerEncoder(
    d_model=512,     # ← 这就是 hidden_dim / embed_dim
    nhead=8,
    dim_feedforward=2048
)

# 我们的 ViT 实现
VisionTransformer(
    embed_dim=128,   # ← 这就是 hidden_dim / d_model
    dim_feedforward=512
)
```

#### 含义 2: Feedforward Hidden Dimension (较少用)

有时 "hidden dimension" 也可能指 **FFN 中间层的维度**。

```python
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        self.fc1 = nn.Linear(embed_dim, hidden_dim)     # 128 → 512
        self.fc2 = nn.Linear(hidden_dim, embed_dim)     # 512 → 128

# 数据流
(N, L, 128) → fc1 → (N, L, 512) → fc2 → (N, L, 128)
                          ↑
                   这里是 512 "hidden"
```

但在我们的代码中，这个维度叫 `dim_feedforward`，所以不会混淆。

---

## 🎓 在 Inline Question 3 中的含义

在题目的上下文中：

```
"Double the hidden dimension"
```

**明确指的是 embed_dim (d_model)**，因为：

1. **题目说明**："Please ignore the computation cost of QKV and output projection"
   - 这说明关注的是 attention 核心计算
   - Attention 的主要维度就是 embed_dim

2. **计算复杂度公式**：
   ```
   Self-Attention Cost = O(L² × D)

   其中 D = embed_dim = hidden_dim
   ```

3. **代码对应**：
   ```python
   model = VisionTransformer(embed_dim=128, ...)  # 原始
   model = VisionTransformer(embed_dim=256, ...)  # "Double hidden dim"
   ```

---

## 📊 完整维度流动示例

```python
# 配置
img_size = 32
patch_size = 8
embed_dim = 128        # ← hidden_dim / d_model
num_heads = 4
dim_feedforward = 512  # ← FFN hidden_dim (不同的 hidden!)

# 数据流
Input Image:           (N, 3, 32, 32)
  ↓ Patch Embedding
Patch Embeddings:      (N, 16, 128)      ← embed_dim 出现
  ↓ Positional Encoding
  ↓ Transformer Encoder Layer 1
    ↓ Multi-Head Attention
      ├─ Split into heads: (N, 4, 16, 32)  ← head_dim = 128/4
      ├─ Attention:        (N, 4, 16, 32)
      └─ Concat:           (N, 16, 128)    ← 回到 embed_dim
    ↓ Add & Norm:          (N, 16, 128)    ← embed_dim 保持
    ↓ Feedforward
      ├─ FC1:              (N, 16, 512)    ← dim_feedforward
      ├─ GELU
      └─ FC2:              (N, 16, 128)    ← 回到 embed_dim
    ↓ Add & Norm:          (N, 16, 128)    ← embed_dim 保持
  ↓ Transformer Encoder Layer 2-6
    ...                    (N, 16, 128)    ← embed_dim 始终保持
  ↓ Global Average Pool
Output Features:         (N, 128)         ← embed_dim
  ↓ Classification Head
Logits:                  (N, 10)
```

**关键观察**：
- `embed_dim = 128` 贯穿整个模型
- 只有在 FFN 内部短暂变成 `dim_feedforward = 512`
- 注意力机制始终工作在 `embed_dim` 空间

---

## 🔑 术语统一表

不同文献/框架使用不同术语，但指的是同一个东西：

| 论文/框架 | 主维度术语 | FFN 中间维度术语 |
|----------|-----------|----------------|
| **原始 Transformer** | d_model | d_ff |
| **BERT** | hidden_size | intermediate_size |
| **GPT** | n_embd | n_inner |
| **ViT 论文** | D | MLP_dim |
| **我们的实现** | embed_dim | dim_feedforward |
| **PyTorch** | d_model | dim_feedforward |
| **Inline Q3** | hidden dimension | - |

**都是在说同一个东西！**

---

## 💡 实用建议

### 如何判断 "hidden dimension" 指什么？

**上下文线索**：

1. **如果在讨论 Attention**：
   ```
   "attention with hidden dimension D"
   → 指 embed_dim
   ```

2. **如果在讨论 FFN**：
   ```
   "feedforward network with hidden dimension H"
   → 可能指 dim_feedforward
   → 但通常会明确说 "feedforward hidden dim"
   ```

3. **如果在讨论整体模型**：
   ```
   "Transformer with hidden dimension 512"
   → 指 embed_dim / d_model
   ```

4. **代码中的参数名**：
   ```python
   # 明确的参数名
   embed_dim=128           # 主维度
   dim_feedforward=512     # FFN 维度

   # 模糊的参数名
   hidden_dim=128          # 通常指 embed_dim
   ```

---

## 📝 回到 Inline Question 3

现在你应该完全理解了：

```
"(i) Double the hidden dimension"
```

**含义**：将 embed_dim 加倍（从 128 → 256）

**影响**：
```python
# 原始
model = VisionTransformer(embed_dim=128, ...)
Cost = O(L² × 128)

# 加倍
model = VisionTransformer(embed_dim=256, ...)
Cost = O(L² × 256) = 2 × O(L² × 128)
```

**为什么不是 dim_feedforward？**

因为题目说了：
> Please ignore the computation cost of QKV and output projection.

这说明只关注 **attention 的核心计算**（softmax 和加权求和），而不是 linear layers。

Attention 的核心计算只涉及 `embed_dim`，不涉及 `dim_feedforward`！

---

## 🎯 总结

| 问题 | 答案 |
|------|------|
| hidden_dim 是 embed_dim 吗？ | **是的**（在大多数情况下）|
| 在 Inline Q3 中？ | **是的**，指 embed_dim |
| 有其他含义吗？ | 有时指 dim_feedforward，但很少见 |
| 如何确定？ | 看上下文和参数名 |

**记住**：当你看到 "hidden dimension" 时，**默认理解为 embed_dim / d_model**，除非上下文明确指出是 FFN 的中间维度。

---

## 📚 参考

```python
# 查看我们的实现
VisionTransformer(
    embed_dim=128,        # ← 这是 "hidden dimension"
    num_heads=4,          # → head_dim = 128/4 = 32
    dim_feedforward=512,  # ← FFN 内部维度
)

# PyTorch 官方
nn.TransformerEncoder(
    d_model=512,          # ← 这是 "hidden dimension"
    nhead=8,
    dim_feedforward=2048
)
```

希望这解释清楚了！🎓
