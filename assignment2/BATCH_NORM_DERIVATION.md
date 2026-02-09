# Batch Normalization Alternative Backward Pass - 详细推导

## 前向传播公式回顾

给定输入 $X = \begin{bmatrix}x_1\\x_2\\\vdots\\x_N\end{bmatrix}$，其中 $x_i \in \mathbb{R}^D$（N个样本，每个样本D个特征）

### 步骤1：计算均值和方差

$$\mu = \frac{1}{N}\sum_{i=1}^N x_i \quad \text{(shape: D)}$$

$$v = \frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2 \quad \text{(shape: D)}$$

### 步骤2：计算标准差

$$\sigma = \sqrt{v + \epsilon} \quad \text{(shape: D)}$$

### 步骤3：标准化

$$\hat{x}_i = \frac{x_i - \mu}{\sigma} \quad \text{(shape: N×D)}$$

### 步骤4：缩放和平移

$$y_i = \gamma \odot \hat{x}_i + \beta \quad \text{(shape: N×D)}$$

---

## 反向传播目标

给定上游梯度 $\frac{\partial L}{\partial Y}$（记为 `dout`），我们需要计算：
- $\frac{\partial L}{\partial X}$（记为 `dx`）
- $\frac{\partial L}{\partial \gamma}$（记为 `dgamma`）
- $\frac{\partial L}{\partial \beta}$（记为 `dbeta`）

---

## 简单梯度的计算

### 1. $\frac{\partial L}{\partial \beta}$

从 $y_i = \gamma \odot \hat{x}_i + \beta$ 可知：

$$\frac{\partial L}{\partial \beta} = \sum_{i=1}^N \frac{\partial L}{\partial y_i} = \sum_{i=1}^N \text{dout}_i$$

在代码中：
```python
dbeta = np.sum(dout, axis=0)  # shape: (D,)
```

### 2. $\frac{\partial L}{\partial \gamma}$

$$\frac{\partial L}{\partial \gamma} = \sum_{i=1}^N \frac{\partial L}{\partial y_i} \odot \hat{x}_i = \sum_{i=1}^N \text{dout}_i \odot \hat{x}_i$$

在代码中：
```python
dgamma = np.sum(dout * x_hat, axis=0)  # shape: (D,)
```

### 3. $\frac{\partial L}{\partial \hat{x}_i}$（中间梯度）

从 $y_i = \gamma \odot \hat{x}_i + \beta$：

$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \odot \gamma = \text{dout}_i \odot \gamma$$

记为 $d\hat{x}_i$：
```python
dx_hat = dout * gamma  # shape: (N, D)
```

---

## 复杂部分：$\frac{\partial L}{\partial X}$ 的推导

这是最复杂的部分，因为 $x_i$ 不仅直接影响 $\hat{x}_i$，还通过 $\mu$ 和 $\sigma$ 间接影响所有的 $\hat{x}_j$。

### 方法一：分步链式法则（原始实现）

从 $\hat{x}_i = \frac{x_i - \mu}{\sigma}$，我们需要考虑三条路径：

#### 路径1：直接影响

$$\frac{\partial \hat{x}_i}{\partial x_i}\bigg|_{\mu,\sigma \text{ const}} = \frac{1}{\sigma}$$

#### 路径2：通过 $\mu$ 的影响

$$\frac{\partial \mu}{\partial x_i} = \frac{1}{N}$$

$$\frac{\partial \hat{x}_j}{\partial \mu} = -\frac{1}{\sigma}$$

$$\frac{\partial L}{\partial \mu} = \sum_{j=1}^N \frac{\partial L}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial \mu} = -\frac{1}{\sigma}\sum_{j=1}^N d\hat{x}_j$$

#### 路径3：通过 $\sigma$ 的影响

首先计算 $\frac{\partial \sigma}{\partial x_i}$：

从 $\sigma = \sqrt{v + \epsilon}$ 和 $v = \frac{1}{N}\sum_k (x_k - \mu)^2$：

$$\frac{\partial v}{\partial x_i} = \frac{2}{N}(x_i - \mu)\left(1 - \frac{1}{N}\right) - \frac{2}{N^2}\sum_{k \neq i}(x_k - \mu)$$

$$= \frac{2}{N}(x_i - \mu) - \frac{2}{N^2}\sum_k(x_k - \mu) = \frac{2}{N}(x_i - \mu)$$

（因为 $\sum_k(x_k - \mu) = 0$）

$$\frac{\partial \sigma}{\partial v} = \frac{1}{2\sqrt{v + \epsilon}} = \frac{1}{2\sigma}$$

$$\frac{\partial \sigma}{\partial x_i} = \frac{\partial \sigma}{\partial v} \cdot \frac{\partial v}{\partial x_i} = \frac{1}{2\sigma} \cdot \frac{2}{N}(x_i - \mu) = \frac{x_i - \mu}{N\sigma} = \frac{\hat{x}_i}{N}$$

$$\frac{\partial \hat{x}_j}{\partial \sigma} = -\frac{x_j - \mu}{\sigma^2} = -\frac{\hat{x}_j}{\sigma}$$

$$\frac{\partial L}{\partial \sigma} = \sum_{j=1}^N \frac{\partial L}{\partial \hat{x}_j} \cdot \frac{\partial \hat{x}_j}{\partial \sigma} = -\frac{1}{\sigma}\sum_{j=1}^N d\hat{x}_j \odot \hat{x}_j$$

#### 组合三条路径

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sigma} + \frac{\partial L}{\partial \mu} \cdot \frac{1}{N} + \frac{\partial L}{\partial \sigma} \cdot \frac{\hat{x}_i}{N}$$

$$= \frac{d\hat{x}_i}{\sigma} - \frac{1}{N\sigma}\sum_j d\hat{x}_j - \frac{\hat{x}_i}{N\sigma}\sum_j (d\hat{x}_j \odot \hat{x}_j)$$

### 方法二：简化公式（Alternative实现）

通过代数简化，可以得到更紧凑的形式：

$$\boxed{\frac{\partial L}{\partial x_i} = \frac{1}{N\sigma}\left[N \cdot d\hat{x}_i - \sum_{j=1}^N d\hat{x}_j - \hat{x}_i \sum_{j=1}^N (d\hat{x}_j \odot \hat{x}_j)\right]}$$

**向量化形式**：

$$\boxed{dX = \frac{1}{N\sigma}\left[N \cdot d\hat{X} - \mathbf{1}_N \sum_{j=1}^N d\hat{x}_j - \hat{X} \sum_{j=1}^N (d\hat{x}_j \odot \hat{x}_j)\right]}$$

其中：
- $\sum_{j=1}^N d\hat{x}_j$ 是沿着axis=0求和，得到shape (D,)
- $\sum_{j=1}^N (d\hat{x}_j \odot \hat{x}_j)$ 是逐元素乘积后沿axis=0求和，得到shape (D,)
- $\mathbf{1}_N$ 表示广播机制，将(D,)扩展为(N,D)

---

## NumPy实现细节

### 简化公式的实现

```python
def batchnorm_backward_alt(dout, cache):
    x_mean, std, x_hat, gamma = cache
    N, D = dout.shape

    # Step 1: 计算 dgamma 和 dbeta
    dgamma = np.sum(dout * x_hat, axis=0)  # (D,)
    dbeta = np.sum(dout, axis=0)           # (D,)

    # Step 2: 计算 dx_hat
    dx_hat = dout * gamma  # (N, D)

    # Step 3: 计算 dx (简化公式)
    # dx = 1/(N*std) * [N*dx_hat - sum(dx_hat) - x_hat*sum(dx_hat*x_hat)]
    sum_dx_hat = np.sum(dx_hat, axis=0)              # (D,) - 第一个求和项
    sum_dx_hat_x_hat = np.sum(dx_hat * x_hat, axis=0)  # (D,) - 第二个求和项

    # 计算括号内的部分
    dx = N * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x_hat  # (N,D)

    # 除以 N*std
    dx = dx / (N * std)  # std会自动广播为 (N,D)

    return dx, dgamma, dbeta
```

---

## 验证正确性

可以通过数值梯度检查验证：
```python
dx_num = eval_numerical_gradient_array(fx, x, dout)
print('dx error: ', rel_error(dx_num, dx))
```

预期误差应该在 1e-13 到 1e-8 之间。

---

## 为什么简化版本更快？

1. **减少中间变量**：不需要显式计算 dstd, dvar, dx_mean 等中间梯度
2. **更好的向量化**：整个计算可以用更少的NumPy操作完成
3. **减少内存访问**：中间结果更少，缓存友好

在实践中，简化版本通常能快 2-4 倍。
