from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        ############################################################################
        # 构建各层维度列表：[input_dim, hidden1, hidden2, ..., num_classes]
        # 例如 input_dim=3072, hidden_dims=[100,50], num_classes=10
        # 则 layer_dims = [3072, 100, 50, 10]
        layer_dims = [input_dim] + hidden_dims + [num_classes]

        for i in range(self.num_layers):
            # 第 i+1 层的权重和偏置（从 1 开始编号：W1, b1, W2, b2, ...）
            self.params[f'W{i+1}'] = weight_scale * np.random.randn(layer_dims[i], layer_dims[i+1])
            self.params[f'b{i+1}'] = np.zeros(layer_dims[i+1])

            # 对隐藏层（非最后一层）添加 batch/layer norm 参数
            # gamma 初始化为 1（缩放因子），beta 初始化为 0（平移因子）
            if self.normalization and i < self.num_layers - 1:
                self.params[f'gamma{i+1}'] = np.ones(layer_dims[i+1])
                self.params[f'beta{i+1}'] = np.zeros(layer_dims[i+1])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        ############################################################################
        # 网络结构：{affine - [batch/layer norm] - relu - [dropout]} x (L-1) - affine
        # caches 字典保存每层的缓存，用于反向传播
        caches = {}
        out = X

        # ---- 前 L-1 层：affine -> [norm] -> relu -> [dropout] ----
        for i in range(1, self.num_layers):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']

            # 仿射变换：out = X * W + b
            out, caches[f'fc{i}'] = affine_forward(out, W, b)

            # 可选的 Batch Normalization 或 Layer Normalization
            if self.normalization == 'batchnorm':
                gamma = self.params[f'gamma{i}']
                beta = self.params[f'beta{i}']
                out, caches[f'bn{i}'] = batchnorm_forward(out, gamma, beta, self.bn_params[i-1])
            elif self.normalization == 'layernorm':
                gamma = self.params[f'gamma{i}']
                beta = self.params[f'beta{i}']
                out, caches[f'ln{i}'] = layernorm_forward(out, gamma, beta, self.bn_params[i-1])

            # ReLU 激活函数
            out, caches[f'relu{i}'] = relu_forward(out)

            # 可选的 Dropout 正则化
            if self.use_dropout:
                out, caches[f'drop{i}'] = dropout_forward(out, self.dropout_param)

        # ---- 最后一层：仅做仿射变换（不加激活函数），输出分类分数 ----
        W = self.params[f'W{self.num_layers}']
        b = self.params[f'b{self.num_layers}']
        scores, caches[f'fc{self.num_layers}'] = affine_forward(out, W, b)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # 步骤 1：计算 softmax 数据损失和对 scores 的梯度
        loss, dscores = softmax_loss(scores, y)

        # 步骤 2：对每一层的权重添加 L2 正则化损失
        # L2 正则化: loss += 0.5 * reg * sum(W^2)
        for i in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * np.sum(self.params[f'W{i}'] ** 2)

        # 步骤 3：反向传播 —— 从最后一层开始逐层回传梯度

        # ---- 最后一层的反向传播（仅 affine）----
        dout, grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = \
            affine_backward(dscores, caches[f'fc{self.num_layers}'])
        # 权重梯度加上正则化项: dW += reg * W
        grads[f'W{self.num_layers}'] += self.reg * self.params[f'W{self.num_layers}']

        # ---- 前 L-1 层的反向传播（顺序：dropout -> relu -> norm -> affine）----
        for i in range(self.num_layers - 1, 0, -1):
            # Dropout 反向传播
            if self.use_dropout:
                dout = dropout_backward(dout, caches[f'drop{i}'])

            # ReLU 反向传播
            dout = relu_backward(dout, caches[f'relu{i}'])

            # Batch/Layer Normalization 反向传播
            if self.normalization == 'batchnorm':
                dout, grads[f'gamma{i}'], grads[f'beta{i}'] = \
                    batchnorm_backward(dout, caches[f'bn{i}'])
            elif self.normalization == 'layernorm':
                dout, grads[f'gamma{i}'], grads[f'beta{i}'] = \
                    layernorm_backward(dout, caches[f'ln{i}'])

            # Affine 反向传播
            dout, grads[f'W{i}'], grads[f'b{i}'] = \
                affine_backward(dout, caches[f'fc{i}'])
            # 权重梯度加上正则化项
            grads[f'W{i}'] += self.reg * self.params[f'W{i}']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
