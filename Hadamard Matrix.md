## Hadamard Matrix 的构造方法

### Sylvester's construction (西尔维斯特构造法)

阿达马矩阵最初的构造的例子是由詹姆斯·西尔维斯特给出的。假设 $H$ 是一个 $n$ 阶的阿达马矩阵, 则下面的矩阵

$$
\left[\begin{array}{cc}
H & H \\
H & -H
\end{array}\right]
$$

给出一个 $2 n$ 阶的阿达马矩阵。连续使用这个方法, 我们可以给出下面的一系列矩阵:

$$
\begin{aligned}
H_1 & =[1] \\
H_2 & =\left[\begin{array}{cc}
1 & 1 \\
1 & -1
\end{array}\right] \\
H_4 & =\left[\begin{array}{cccc}
1 & 1 & 1 & 1 \\
1 & -1 & 1 & -1 \\
1 & 1 & -1 & -1 \\
1 & -1 & -1 & 1
\end{array}\right] \\
\vdots &
\end{aligned}
$$

利用这种方法, 西尔维斯特成功地构造了任何 $2^k$ 阶阿达马矩阵, 其中 $k$ 为非负整数。
西尔维斯特给出的矩阵有些特殊的性质。他们都是对称矩阵, 并且这些矩阵的迹都是 0 。第一行和第一列的元素都是 +1 , 其他各行各列的元素都是一半+1，一半-1。这些矩阵和沃尔什函数有密切的关系。

### Alternative construction (替代构造法)

如果我们通过群同态将哈达玛矩阵的元素从 $\{1,-1, \times\}$ 映射到 $\{0,1, \oplus\}$，我们可以描述一种Sylvester's Hadamard矩阵的替代构造方法。
首先，考虑矩阵 $F_n$ ，这是一个 $n \times 2^n$ 的矩阵，其列由所有按升序排列的 $n$ 位数构成。我们可以递归地定义 $F_n$ ，如下：

$$
\begin{aligned}
F_1 & =\left[\begin{array}{ll}
0 & 1
\end{array}\right] \\
F_n & =\left[\begin{array}{cc}
0_{1 \times 2^{n-1}} & 1_{1 \times 2^{n-1}} \\
F_{n-1} & F_{n-1}
\end{array}\right] .
\end{aligned}
$$

可以通过归纳法证明，哈达玛矩阵在上述同态下的像由以下公式给出

$$
H_{2^n}=F_n^{\top} F_n .
$$

这个构造法表明，哈达玛矩阵 $H_{2^n}$ 的行可以被视为一个长度为 $2^n$ 的线性纠错码，秩为 $n$ ，最小距离为 $2^{n-1}$ ，生成矩阵为 $F_n$ 。
这种码也被称为Walsh码。相比之下，哈达玛码是通过稍微不同的程序从哈达玛矩阵 $H_{2^n}$ 中构造的。

## Hadamard Fixed Classifer

### 原论文的定义：https://arxiv.org/abs/1801.04540

> We further suggest the use of a Hadamard matrix (Hedayat et al., 1978) as the final classification transform. Hadamard matrix $H$ is an $n \times n$ matrix, where all of its entries are either +1 or -1 . Further more, $H$ is orthogonal, such that $H H^T=n I_n$ where $I_n$ is the identity matrix.
> 
> We can use a truncated Hadamard matrix $\hat{H} \in\{-1,1\}^{C \times N}$ where all $C$ rows are orthogonal as our final classification layer such that
> 
> $$
> y=\hat{H} \hat{x}+b
> $$
> 
> This usage allows two main benefits:
> 
> - A deterministic, low-memory and easily generated matrix that can be used to classify.
> - 
> - Removal of the need to perform a full matrix-matrix multiplication - as multiplying by a Hadamard matrix can be done by simple sign manipulation and addition.
> 
> We note that $n$ must be a multiple of 4 , but it can be easily truncated to fit normally defined networks. We also note the similarity of using a Hadamard matrix as a final classifier to methods of weight binarization such as the one suggested by Courbariaux et al. (2015). As the classifier weights are fixed to need only 1-bit precision, it is now possible to focus our attention on the features preceding it.

以下是原文的中文翻译版：

> 我们进一步建议使用哈达玛矩阵（Hedayat等，1978）作为最终的分类转换。哈达玛矩阵 $H$ 是一个 $n \times n$ 的矩阵，其中所有的元素都是+1或-1。此外， $H$ 是正交的，因此 $HH^T = nI_n$ ，其中 $I_n$ 是单位矩阵。
> 
> 我们可以使用截断的哈达玛矩阵 $\hat{H} \in\{-1,1\}^{C \times N}$ ，其中所有的 $C$ 行都是正交的，作为我们最终的分类层，即
> 
> $$
> y=\hat{H} \hat{x}+b
> $$
> 
> 这种使用带来两个主要的好处：
> 
> - 一个确定的、低内存且容易生成的矩阵，可以用来分类。
> - 
> - 消除了执行全矩阵-矩阵乘法的需要——因为通过哈达玛矩阵的乘法可以通过简单的符号操作和加法完成。
> 
> 我们注意到， $n$ 必须是4的倍数，但它可以很容易地被截断以适应通常定义的网络。
> 
> 我们还注意到，使用哈达玛矩阵作为最终的分类器与权重二值化方法的相似性，如Courbariaux等人（2015）所建议的。由于分类器权重被固定为只需要1位精度，我们现在可以将注意力集中在其前面的特征上。

以下是原文提供的实现Hadamard Fixed Classifer的代码：

```Python
import torch.nn as nn
import math
import torch
from torch.autograd import Variable
from scipy.linalg import hadamard

class HadamardProj(nn.Module):

    def __init__(self, input_size, output_size, bias=True, fixed_weights=True, fixed_scale=None):
        super(HadamardProj, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        sz = 2 ** int(math.ceil(math.log(max(input_size, output_size), 2)))
        mat = torch.from_numpy(hadamard(sz))
        if fixed_weights:
            self.proj = Variable(mat, requires_grad=False)
        else:
            self.proj = nn.Parameter(mat)

        init_scale = 1. / math.sqrt(self.output_size)

        if fixed_scale is not None:
            self.scale = Variable(torch.Tensor(
                [fixed_scale]), requires_grad=False)
        else:
            self.scale = nn.Parameter(torch.Tensor([init_scale]))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(
                output_size).uniform_(-init_scale, init_scale))
        else:
            self.register_parameter('bias', None)

        self.eps = 1e-8

    def forward(self, x):
        if not isinstance(self.scale, nn.Parameter):
            self.scale = self.scale.type_as(x)
        x = x / (x.norm(2, -1, keepdim=True) + self.eps)
        w = self.proj.type_as(x)

        out = -self.scale * \
            nn.functional.linear(x, w[:self.output_size, :self.input_size])
        if self.bias is not None:
            out = out + self.bias.view(1, -1)
        return out

```

以下是Chatgbt对这段代码的解释：

> 这段代码实现了一个名为`HadamardProj`的PyTorch模块，该模块将输入进行哈达玛矩阵投影，用于在神经网络中实现变换。以下是代码的详细解释：
> 
> - `__init__`函数：初始化模块。接收输入尺寸，输出尺寸，是否包含偏置，是否使用固定权重，以及固定比例等参数。首先，根据输入和输出尺寸的最大值，计算哈达玛矩阵的大小。然后，创建该哈达玛矩阵。如果`fixed_weights`参数为True，则矩阵的权重将固定不变，否则，权重将作为需要学习的参数。接下来，设置比例参数。如果`fixed_scale`参数提供了值，则比例将被固定，否则，比例将被初始化为`1 / sqrt(output_size)`，并作为需要学习的参数。如果`bias`参数为True，那么将创建偏置参数，初始化为在`-init_scale`和`init_scale`之间均匀分布的随机值。最后，设置一个非常小的正值`eps`，用于防止除以0的情况。
> 
> - `forward`函数：定义前向传播过程。首先，确保比例参数和输入数据的数据类型一致。然后，将输入数据进行归一化，以保证L2范数为1。接着，将输入数据和截取的哈达玛矩阵进行线性变换，并乘以比例参数，取负值作为输出。如果存在偏置，那么还要加上偏置。最后，返回输出数据。
> 
> 这个模块的用途主要是为了实现哈达玛投影，用于将输入数据转换到不同的空间中，以便进行后续的计算。哈达玛投影的特性使得这个过程能够有效地处理大量数据，并且保持较低的计算复杂性。

以下是对代码中几个关键函数进行说明：

```Python
from scipy.linalg import hadamard
sz = 2 ** int(math.ceil(math.log(max(input_size, output_size), 2)))
mat = torch.from_numpy(hadamard(sz))
```
这段代码首先计算了一个接近于`max(input_size, output_size)`的最小的2的幂`sz`，然后使用SciPy的`hadamard`函数生成了一个`sz`x`sz`的哈达玛矩阵，最后使用`torch.from_numpy`将这个哈达玛矩阵从NumPy数组转换为PyTorch的张量。

以下是这段代码的详细步骤：

1. `sz = 2 ** int(math.ceil(math.log(max(input_size, output_size), 2)))`：这行代码首先计算`input_size`和`output_size`中的最大值，然后求其以2为底的对数，并向上取整，得到一个整数。最后，求2的该整数次幂，得到`sz`。这样得到的`sz`就是接近于`max(input_size, output_size)`的最小的2的幂。

2. `mat = torch.from_numpy(hadamard(sz))`：这行代码首先使用`hadamard(sz)`生成一个`sz`x`sz`的哈达玛矩阵。然后，使用`torch.from_numpy`将这个哈达玛矩阵从NumPy数组转换为PyTorch的张量，赋值给`mat`。

需要注意的是，哈达玛矩阵的大小必须是2的幂，这就是在计算`sz`时需要取2的幂的原因。


