# MultiHeadAttention 实现分析

## 标准 MultiHeadAttention 实现步骤

1. **线性映射**：将输入的 query、key、value 通过各自的线性层映射到不同的表示空间
2. **分割多头**：将映射后的表示分割成多个头
3. **计算注意力**：对每个头计算注意力分数、应用掩码、softmax 归一化
4. **合并多头**：将所有头的结果合并
5. **最终投影**：通过最终的线性层输出结果

## 当前实现分析

### 代码实现（第 158-165 行）

```python
H = self.n_head
# E = QK / square of D
query = query.reshape(N, S, H, E//H).transpose(1,2) # N, H, S, E//H
key = key.reshape(N, T, H, E//H).permute(0,2,3,1)  # N, H, E//H, T
e = attn_mask * torch.matmul(query,key).divide(torch.sqrt(S)) # N, H, S, T
a = torch.softmax(e,dim=(2,3))
value = value.reshape(N, T, H, E//H).transpose(1,2)   # N, H, T, E//H
output = torch.matmul(a,value).reshape(N, S, E)
```

### 问题分析

1. **缺少线性映射**：
   - 标准实现中，query、key、value 应该先通过 `self.query`、`self.key`、`self.value` 线性层进行映射
   - 当前代码直接对输入的 query、key、value 进行了 reshape 操作

2. **缩放因子错误**：
   - 代码中使用了 `torch.sqrt(S)` 作为缩放因子
   - 正确的缩放因子应该是 `torch.sqrt(E//H)`（每个头的维度的平方根）

3. **注意力掩码应用方式错误**：
   - 代码中使用了 `attn_mask * torch.matmul(...)`
   - 正确的做法应该是使用 `masked_fill` 将掩码位置的值设置为 `-inf`，然后再应用 softmax

4. **缺少最终投影层**：
   - 标准实现中，合并所有头的结果后应该通过 `self.proj` 线性层进行投影
   - 当前代码直接返回了合并后的结果

5. **缺少注意力 dropout**：
   - 代码中定义了 `self.attn_drop`，但没有在计算注意力权重后应用它

6. **softmax 维度错误**：
   - 代码中使用了 `dim=(2,3)`
   - 正确的做法应该是只在最后一个维度（即 target sequence length 维度）上应用 softmax，即 `dim=3`

## 正确实现示例

```python
def forward(self, query, key, value, attn_mask=None):
    N, S, E = query.shape
    N, T, E = value.shape
    
    # 1. 线性映射
    query = self.query(query)
    key = self.key(key)
    value = self.value(value)
    
    # 2. 分割多头
    H = self.n_head
    query = query.reshape(N, S, H, E//H).transpose(1, 2)  # N, H, S, E//H
    key = key.reshape(N, T, H, E//H).permute(0, 2, 3, 1)  # N, H, E//H, T
    value = value.reshape(N, T, H, E//H).transpose(1, 2)  # N, H, T, E//H
    
    # 3. 计算注意力
    attn_scores = torch.matmul(query, key) / torch.sqrt(torch.tensor(E//H, dtype=torch.float32))
    
    # 应用掩码
    if attn_mask is not None:
        attn_scores = attn_scores.masked_fill(attn_mask == 0, -float('inf'))
    
    # softmax 归一化
    attn_weights = torch.softmax(attn_scores, dim=-1)
    
    # 应用 dropout
    attn_weights = self.attn_drop(attn_weights)
    
    # 4. 合并多头
    output = torch.matmul(attn_weights, value)  # N, H, S, E//H
    output = output.transpose(1, 2).reshape(N, S, E)  # N, S, E
    
    # 5. 最终投影
    output = self.proj(output)
    
    return output
```

## 缩放因子的作用与选择

### 为什么使用每个头的维度的平方根作为缩放因子？

在 Transformer 的 MultiHeadAttention 中，使用每个头的维度的平方根（即 \(\sqrt{d_k}\)，其中 \(d_k = E/H\) 是每个头的维度）作为缩放因子，主要是为了**控制注意力分数的方差**，避免数值不稳定问题：

1. **注意力分数的计算**：标准的注意力机制通过 query（查询）和 key（键）的点积来计算相似度：
   \[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]

2. **方差分析**：假设 query 和 key 的元素都是独立同分布的随机变量，均值为 0，方差为 1。那么它们的点积的方差会随着维度 \(d_k\) 的增加而线性增长：
   - 单个元素 \(q_i \cdot k_i\) 的方差为 1
   - 长度为 \(d_k\) 的向量点积的方差为 \(d_k\)

3. **缩放因子的作用**：通过除以 \(\sqrt{d_k}\)，可以将点积的方差归一化到 1，避免以下问题：
   - **数值爆炸**：当 \(d_k\) 较大时，点积的绝对值会变得很大
   - **梯度消失**：softmax 函数在输入值差异很大时，会将概率集中到最大值对应的位置，导致其他位置的梯度接近 0

4. **为什么是平方根**：
   \[ \text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{1}{d_k} \cdot \text{Var}(q \cdot k) = \frac{1}{d_k} \cdot d_k = 1 \]

## 维度定义与区别

在 Transformer 的 MultiHeadAttention 中，各维度的定义如下：

| 变量 | 名称 | 描述 |
|------|------|------|
| \(N\) | Batch Size | 单个批次中处理的样本数 |
| \(S\) | Source Sequence Length | 作为 **query** 的输入序列长度（如解码器输入） |
| \(T\) | Target Sequence Length | 作为 **key/value** 的输入序列长度（如编码器输出） |
| \(E\) | Embedding Dimension | 每个 token 的向量表示大小（固定的超参数） |

### 核心区别
- **\(E\) vs. \(S\)/\(T\)**：
  - \(E\) 是**特征维度**（如 512、768），定义了每个 token 向量表示的大小，是固定的超参数
  - \(S\) 和 \(T\) 是**序列长度维度**（如 10、50），表示输入序列中的 token 数量，是动态的，取决于实际输入数据

## 结论

当前的 MultiHeadAttention 实现存在多个问题，与标准的 Transformer 架构不符。主要问题包括缺少线性映射、缩放因子错误、注意力掩码应用方式错误、缺少最终投影层、缺少注意力 dropout 以及 softmax 维度错误。

建议按照上述正确实现示例进行修改，以确保 MultiHeadAttention 的功能正确。