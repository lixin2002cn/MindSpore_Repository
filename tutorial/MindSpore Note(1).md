# MindSpore Note(1)

## 1. 安装MindSpore

官网链接：https://www.mindspore.cn/install/

## 2. 快速入门

代码示例来源于官网：https://www.mindspore.cn/tutorials/zh-CN/master/beginner/quick_start.html#

```python
import mindspore
from mindspore import nn
from mindspore.dataset import vision,transforms
from mindspore.dataset import MnistDataset
from download import download 

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)

# get the dataset object 
train_dataset = MnistDataset('MNIST_Data/train')
test_dataset = MnistDataset('MNIST_Data/test')

# print the cols of dataset for data prosessing
print(train_dataset.get_col_names())

# set the data pipeline 


def datapipe(dataset, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

# Map vision transforms and batch dataset
train_dataset = datapipe(train_dataset, 64)
test_dataset = datapipe(test_dataset, 64)

# check the shape, dtype of dataset
for image, label in test_dataset.create_tuple_iterator():
    print(f"Shape of image [N, C, H, W]: {image.shape} {image.dtype}")
    print(f"Shape of label: {label.shape} {label.dtype}")
    break

# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()
print(model)

# train pipeline
# Instantiate loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)

# 1. Define forward function
def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits

# 2. Get gradient function
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

# 3. Define function of one-step training
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss

def train(model, dataset):
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")

# Save checkpoint
mindspore.save_checkpoint(model, "model.ckpt")
print("Saved Model to model.ckpt")


# Instantiate a random initialized model
model = Network()
# Load checkpoint and load parameter to model
param_dict = mindspore.load_checkpoint("model.ckpt")
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
print(param_not_load)
model.set_train(False)
for data, label in test_dataset:
    pred = model(data)
    predicted = pred.argmax(1)
    print(f'Predicted: "{predicted[:10]}", Actual: "{label[:10]}"')
    break

    

```

## 3. 张量Tensor

### 3.1 创建张量

1. 根据数据直接生成
2. 从numpy数组生成
3. 使用init初始化器构造张量
4. 继承另一个张量的属性，形成新的张量

```python
import numpy as np
import mindspore
from mindspore import ops
from mindspore import Tensor, CSRTensor, COOTensor

data = [1, 0, 1, 0]
x_data = Tensor(data)
print(x_data, x_data.shape, x_data.dtype)

np_array = np.array(data)
x_np = Tensor(np_array)
print(x_np, x_np.shape, x_np.dtype)

from mindspore.common.initializer import One, Normal

# Initialize a tensor with ones
tensor1 = mindspore.Tensor(shape=(2, 2), dtype=mindspore.float32, init=One())
# Initialize a tensor from normal distribution
tensor2 = mindspore.Tensor(shape=(2, 2), dtype=mindspore.float32, init=Normal())

print("tensor1:\n", tensor1)
print("tensor2:\n", tensor2)


from mindspore import ops

x_ones = ops.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_zeros = ops.zeros_like(x_data)
print(f"Zeros Tensor: \n {x_zeros} \n")


```

### 3.2 张量操作

类似于numpy数组切片

```python
tensor = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))

print("First row: {}".format(tensor[0]))
print("value of bottom right corner: {}".format(tensor[1, 1]))
print("Last column: {}".format(tensor[:, -1]))
print("First column: {}".format(tensor[..., 0]))

```

常见张量运算有：

```python
x = Tensor(np.array([1, 2, 3]), mindspore.float32)
y = Tensor(np.array([4, 5, 6]), mindspore.float32)

output_add = x + y
output_sub = x - y
output_mul = x * y
output_div = y / x
output_mod = y % x
output_floordiv = y // x

print("add:", output_add)
print("sub:", output_sub)
print("mul:", output_mul)
print("div:", output_div)
print("mod:", output_mod)
print("floordiv:", output_floordiv)

data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.concat((data1, data2), axis=0)

print(output)
print("shape:\n", output.shape)

data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.stack([data1, data2])

print(output)
print("shape:\n", output.shape)


```

与numpy数组互相转化

```python
t = Tensor([1., 1., 1., 1., 1.])
print(f"t: {t}", type(t))
n = t.asnumpy()
print(f"n: {n}", type(n))

n = np.ones(5)
t = Tensor.from_numpy(n)


```

## 4 数据集Dataset

### 4.1 自定义数据集

```python
# Random-accessible object as input source
class RandomAccessDataset:
    def __init__(self):
        self._data = np.ones((5, 2))
        self._label = np.zeros((5, 1))

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)
loader = RandomAccessDataset()
dataset = GeneratorDataset(source=loader, column_names=["data", "label"])

for data in dataset:
    print(data)

# list, tuple are also supported.
loader = [np.array(0), np.array(1), np.array(2)]
dataset = GeneratorDataset(source=loader, column_names=["data"])

for data in dataset:
    print(data)

```

### 4.2 可迭代数据集

```python
# Iterator as input source
class IterableDataset():
    def __init__(self, start, end):
        '''init the class object to hold the data'''
        self.start = start
        self.end = end
    def __next__(self):
        '''iter one data and return'''
        return next(self.data)
    def __iter__(self):
        '''reset the iter'''
        self.data = iter(range(self.start, self.end))
        return self

# Generator
def my_generator(start, end):
    for i in range(start, end):
        yield i

```

## 5 数据变换Transforms

```python
import numpy as np
from PIL import Image
from download import download
from mindspore.dataset import transforms, vision, text
from mindspore.dataset import GeneratorDataset, MnistDataset

composed = transforms.Compose(
    [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
)

train_dataset = train_dataset.map(composed, 'image')
image, label = next(train_dataset.create_tuple_iterator())
print(image.shape)



```

## 6 网络构建

```python
import mindspore
from mindspore import nn, ops

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 10, weight_init="normal", bias_init="zeros")
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()
print(model)

X = ops.ones((1, 28, 28), mindspore.float32)
logits = model(X)
# print logits
logits

pred_probab = nn.Softmax(axis=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


print(f"Model structure: {model}\n\n")

for name, param in model.parameters_and_names():
    print(f"Layer: {name}\nSize: {param.shape}\nValues : {param[:2]} \n")

```

## 7 自动微分

```python
import numpy as np
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter

x = ops.ones(5, mindspore.float32)  # input tensor
y = ops.zeros(3, mindspore.float32)  # expected output
w = Parameter(Tensor(np.random.randn(5, 3), mindspore.float32), name='w') # weight
b = Parameter(Tensor(np.random.randn(3,), mindspore.float32), name='b') # bias

def function(x, y, w, b):
    z = ops.matmul(x, w) + b
    loss = ops.binary_cross_entropy_with_logits(z, y, ops.ones_like(z), ops.ones_like(z))
    return loss

loss = function(x, y, w, b)
print(loss)

grad_fn = mindspore.grad(function, (2, 3))

grads = grad_fn(x, y, w, b)
print(grads)

def function_with_logits(x, y, w, b):
    z = ops.matmul(x, w) + b
    loss = ops.binary_cross_entropy_with_logits(z, y, ops.ones_like(z), ops.ones_like(z))
    return loss, z
grad_fn = mindspore.grad(function_with_logits, (2, 3))
grads = grad_fn(x, y, w, b)
print(grads)

def function_stop_gradient(x, y, w, b):
    z = ops.matmul(x, w) + b
    loss = ops.binary_cross_entropy_with_logits(z, y, ops.ones_like(z), ops.ones_like(z))
    return loss, ops.stop_gradient(z)

grad_fn = mindspore.grad(function_stop_gradient, (2, 3))
grads = grad_fn(x, y, w, b)
print(grads)


```

## 8 训练流程

```python
# Define forward function
def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits

# Get gradient function
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

# Define function of one-step training
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss

def train_loop(model, dataset):
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
            
def test_loop(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), learning_rate=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(model, train_dataset)
    test_loop(model, test_dataset, loss_fn)
print("Done!")

```

## 9 保存和加载

```python
import numpy as np
import mindspore
from mindspore import nn
from mindspore import Tensor


def network():
    model = nn.SequentialCell(
                nn.Flatten(),
                nn.Dense(28*28, 512),
                nn.ReLU(),
                nn.Dense(512, 512),
                nn.ReLU(),
                nn.Dense(512, 10))
    return model

model = network()
mindspore.save_checkpoint(model, "model.ckpt")

model = network()
param_dict = mindspore.load_checkpoint("model.ckpt")
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
print(param_not_load)

mindspore.set_context(mode=mindspore.GRAPH_MODE, jit_syntax_level=mindspore.STRICT)
model = network()
inputs = Tensor(np.ones([1, 1, 28, 28]).astype(np.float32))
mindspore.export(model, inputs, file_name="model", file_format="MINDIR")

graph = mindspore.load("model.mindir")
model = nn.GraphCell(graph)
outputs = model(inputs)
print(outputs.shape)


```

## 10 图模式

动态图的特点是计算图的构建和计算同时发生（Define by run），其符合Python的解释执行方式，在计算图中定义一个Tensor时，其值就已经被计算且确定，因此在调试模型时较为方便，能够实时得到中间结果的值，但由于所有节点都需要被保存，导致难以对整个计算图进行优化。

在MindSpore中，动态图模式又被称为PyNative模式。由于动态图的解释执行特性，在脚本开发和网络流程调试过程中，推荐使用动态图模式进行调试。 如需要手动控制框架采用PyNative模式，可以通过以下代码进行网络构建：

```python
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
ms.set_context(mode=ms.PYNATIVE_MODE)  # 使用set_context进行动态图模式的配置

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()
input = Tensor(np.ones([64, 1, 28, 28]).astype(np.float32))
output = model(input)
print(output)

```

在MindSpore中，静态图模式又被称为Graph模式，在Graph模式下，基于图优化、计算图整图下沉等技术，编译器可以针对图进行全局的优化，获得较好的性能，因此比较适合网络固定且需要高性能的场景。

如需要手动控制框架采用静态图模式，可以通过以下代码进行网络构建：

```python
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
ms.set_context(mode=ms.GRAPH_MODE)  # 使用set_context进行运行静态图模式的配置

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()
input = Tensor(np.ones([64, 1, 28, 28]).astype(np.float32))
output = model(input)
print(output)

```

通常情况下，由于动态图的灵活性，我们会选择使用PyNative模式来进行自由的神经网络构建，以实现模型的创新和优化。但是当需要进行性能加速时，我们需要对神经网络部分或整体进行加速。MindSpore提供了两种切换为图模式的方式，分别是基于装饰器的开启方式以及基于全局context的开启方式。

### 基于装饰器的开启方式

MindSpore提供了jit装饰器，可以通过修饰Python函数或者Python类的成员函数使其被编译成计算图，通过图优化等技术提高运行速度。此时我们可以简单的对想要进行性能优化的模块进行图编译加速，而模型其他部分，仍旧使用解释执行方式，不丢失动态图的灵活性。无论全局context是设置成静态图模式还是动态图模式，被jit修饰的部分始终会以静态图模式进行运行。

在需要对Tensor的某些运算进行编译加速时，可以在其定义的函数上使用jit修饰器，在调用该函数时，该模块自动被编译为静态图。需要注意的是，jit装饰器只能用来修饰函数，无法对类进行修饰。jit的使用示例如下：

```python
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

input = Tensor(np.ones([64, 1, 28, 28]).astype(np.float32))

@ms.jit  # 使用ms.jit装饰器，使被装饰的函数以静态图模式运行
def run(x):
    model = Network()
    return model(x)

output = run(input)
print(output)


import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

input = Tensor(np.ones([64, 1, 28, 28]).astype(np.float32))

def run(x):
    model = Network()
    return model(x)

run_with_jit = ms.jit(run)  # 通过调用jit将函数转换为以静态图方式执行
output = run_with_jit(input)
print(output)

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    @ms.jit  # 使用ms.jit装饰器，使被装饰的函数以静态图模式运行
    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

input = Tensor(np.ones([64, 1, 28, 28]).astype(np.float32))
model = Network()
output = model(input)
print(output)


```

