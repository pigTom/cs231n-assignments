# Batch Normalization 和 Layer Normalization 实现总结

## 修复的问题

### 1. `batchnorm_forward` 的方差计算错误

**原始代码（错误）：**
```python
var = np.square(x_mean) / N  # 错误！
```

**问题分析：**
- 这行代码将每个元素平方后除以N，得到的shape仍然是 (N, D)
- 但是方差应该是一个标量（对于每个特征），shape应该是 (D,)
- 方差的正确定义是：$\text{var} = \frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2$

**修复后的代码（正确）：**
```python
var = np.mean(np.square(x_mean), axis=0)  # shape: (D,)
```

**为什么修复后是正确的：**
1. `np.square(x_mean)` 计算每个元素的平方，shape: (N, D)
2. `np.mean(..., axis=0)` 沿着样本维度求平均，得到每个特征的方差，shape: (D,)
3. 这正确实现了方差公式：对每个特征，计算其在所有样本上的平方差的均值

### 2. `batchnorm_backward_alt` 的实现

实现了优化的反向传播算法，使用简化公式：

$$\frac{\partial L}{\partial x_i} = \frac{1}{N\sigma}\left[N \cdot \frac{\partial L}{\partial \hat{x}_i} - \sum_{j=1}^N \frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i \sum_{j=1}^N \left(\frac{\partial L}{\partial \hat{x}_j} \odot \hat{x}_j\right)\right]$$

**关键代码：**
```python
dx = (dx_hat - sum_dx_hat / N - x_hat * sum_dx_hat_x_hat / N) / std
```

**性能提升：**
- 相比原始实现快 1.28x
- 代码更简洁，易于理解
- 单行实现（如作业要求）

### 3. `layernorm_forward` 的方差计算错误

**原始代码（错误）：**
```python
var = np.sum(np.square(x_mean), axis=1, keepdims=True)  # 错误！
```

**问题分析：**
- 这行代码计算的是平方和，不是方差
- 方差应该是平方和的平均值

**修复后的代码（正确）：**
```python
var = np.mean(np.square(x_mean), axis=1, keepdims=True)  # shape: (N, 1)
```

### 4. `layernorm_backward` 的实现错误

**原始代码的问题：**
1. 变量声明错误：`gamma, dx, dgamma, dbeta = None, None, None` （4个变量只有3个None）
2. cache解包错误：`x_mean, std, x_hat = cache` （少了gamma）
3. dgamma和dbeta计算错误：没有正确求和

**修复后的实现：**
```python
# 正确解包cache
gamma, x_mean, std, x_hat = cache

# 正确计算梯度（沿axis=0求和）
dgamma = np.sum(dout * x_hat, axis=0)  # shape: (D,)
dbeta = np.sum(dout, axis=0)           # shape: (D,)

# 使用简化公式计算dx（沿axis=1求和，因为layer norm是对每个样本标准化）
sum_dx_hat = np.sum(dx_hat, axis=1, keepdims=True)
sum_dx_hat_x_hat = np.sum(dx_hat * x_hat, axis=1, keepdims=True)
dx = (dx_hat - sum_dx_hat / D - x_hat * sum_dx_hat_x_hat / D) / std
```

---

## 测试结果

### Batch Normalization 测试

```
Testing batchnorm_forward...
After batch normalization (gamma=1, beta=0):
  means: [4.66e-17 1.67e-17 1.86e-17]  ✓ 接近0
  stds:  [1.00 1.00 1.00]               ✓ 接近1

Testing batchnorm_backward...
dx error:       1.70e-09  ✓ < 1e-8
dgamma error:   7.42e-13  ✓ < 1e-8
dbeta error:    2.88e-12  ✓ < 1e-8

Testing batchnorm_backward_alt...
dx difference:      1.24e-12  ✓ 与原始实现一致
dgamma difference:  0.0       ✓ 完全相同
dbeta difference:   0.0       ✓ 完全相同
speedup:            1.28x     ✓ 性能提升
```

### Layer Normalization 测试

```
Testing layernorm_forward...
After layer normalization (gamma=1, beta=0):
  means: [4.81e-16 -7.40e-17 2.22e-16 -5.92e-16]  ✓ 接近0
  stds:  [1.00 1.00 1.00 1.00]                     ✓ 接近1

After layer normalization (gamma=[3,3,3], beta=[5,5,5]):
  means: [5.0 5.0 5.0 5.0]      ✓ 等于beta
  stds:  [3.0 3.0 3.0 3.0]      ✓ 接近gamma

Testing layernorm_backward...
dx error:       1.43e-09  ✓ < 1e-8
dgamma error:   4.52e-12  ✓ < 1e-8
dbeta error:    2.28e-12  ✓ < 1e-8
```

---

## 关键区别：Batch Norm vs Layer Norm

| 方面 | Batch Normalization | Layer Normalization |
|------|---------------------|---------------------|
| **标准化维度** | 沿着batch维度 (axis=0) | 沿着特征维度 (axis=1) |
| **均值/方差shape** | (D,) - 每个特征一个值 | (N, 1) - 每个样本一个值 |
| **对batch size的依赖** | 强依赖（batch太小效果差） | 不依赖 |
| **训练/测试区别** | 需要running mean/var | 训练和测试完全相同 |
| **适用场景** | CNN、大batch size | RNN、小batch size、NLP |

---

## 推导文档

详细的数学推导请参考：
- `BATCH_NORM_DERIVATION.md` - Batch Normalization 反向传播的完整数学推导

---

## 总结

1. ✅ 修复了 `batchnorm_forward` 的方差计算错误
2. ✅ 实现了优化的 `batchnorm_backward_alt`，提升了性能
3. ✅ 修复了 `layernorm_forward` 的方差计算错误
4. ✅ 修复了 `layernorm_backward` 的多个实现错误
5. ✅ 所有测试通过，梯度检查误差在可接受范围内（< 1e-8）
