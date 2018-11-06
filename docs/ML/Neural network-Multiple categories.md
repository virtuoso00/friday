## 利用多层神经网络解决手写数字图像识别的多分类问题





### 多分类神经网络算法流程（BP算法流程）：

在手工设定了神经网络的层数，每层神经元的个数，学习率 η 后，BP 算法会先随机初始化每条连接线权重和偏置，然后对于训练集中的每个输入 X 和输出 y，BP 算法都会先执行前向传输得到预测值，然后根据真实值与预测值之间的误差执行逆向反馈更新神经网络中每条连接线的权重和每层的偏好。在没有到达停止条件的情况下重复上述过程。

##### 停止的条件有下面三种：

- 权重的更新低于某个阀值时
- 达到预设的迭代次数（我们这次试验选择了这种停止条件）
- 预测的错误率低于某个阀值

 每输入一个实例，神经网络会执行前向输出一层层计算到输出神经元的值，通过过哪一个输出神经元的值最大来预测输入的实例所代表的数字。然后根据输出神经元的值，计算出预测值与真实值之间的误差，再逆向反馈更新神经网络中每条连接线的权值和每个神经元的偏好。

#### 前向传输（Feed-Forward）

输入层-->隐藏层-->输出层逐层的计算所有神经元输出值的过程。

#### 误差逆传输（error BackPropagation）

 由于输出层的值与真实值会存在误差，用均方误差来衡量预测值与真实值之间的误差

![1541230932231](C:\Users\LRIDES~1\AppData\Local\Temp\1541230932231.png)

 逆向反馈的目标就是让E函数的值尽可能的小，而每个神经元的输出值是由该点的连接线对应的权重值和该层对应的偏好所决定的，因此，要让误差函数达到最小，我们就要调整w和b值， 使得误差函数的值最小。 

![1541231067549](C:\Users\LRIDES~1\AppData\Local\Temp\1541231067549.png)

 





对目标函数 E 求 w 和 b 的偏导可以得到 w 和 b 的更新量，下面拿求 w 偏导来做推导 

![1541231176817](C:\Users\LRIDES~1\AppData\Local\Temp\1541231176817.png)

其中 η 为学习率，取值通常为 0.1 ~ 0.3,可以理解为每次梯度所迈的步伐。注意到 w_hj 的值先影响到第 j 个输出层神经元的输入值a，再影响到输出值y，根据链式求导法则有 

![1541231218514](C:\Users\LRIDES~1\AppData\Local\Temp\1541231218514.png)

根据神经元输出值 a 的定义有 

![1541231294555](C:\Users\LRIDES~1\AppData\Local\Temp\1541231294555.png)

Sigmoid 求导数的式子如下 ：

![1541231329144](C:\Users\LRIDES~1\AppData\Local\Temp\1541231329144.png)

因而：

![1541231371846](C:\Users\LRIDES~1\AppData\Local\Temp\1541231371846.png)



则权重 w 的更新量为 ：

![1541231421542](C:\Users\LRIDES~1\AppData\Local\Temp\1541231421542.png)







同理可得b的更新量为：

![1541231471761](C:\Users\LRIDES~1\AppData\Local\Temp\1541231471761.png)

但这两个公式只能够更新输出层与前一层连接线的权重和输出层的偏置，原因是因为 δ 值依赖了真实值y这个变量，但是我们只知道输出层的真实值而不知道每层隐藏层的真实值，导致无法计算每层隐藏层的 δ 值，所以我们希望能够利用 **i+1** 层的 δ 值来计算 **i** 层的 δ 值，而恰恰通过一些列数学转换后可以做到，这也就是逆向反馈名字的由来，公式如下:

![1541231542232](C:\Users\LRIDES~1\AppData\Local\Temp\1541231542232.png)

 从上面的式子，只需要知道下一层的权重的神经元输出层的值就可以计算出上一层的 δ 值 ，只要通过不断地利用上面的式子就可以就可以更新隐藏层的全部权重和偏置。

### 下面给出此次实验的源码：

##### BPNetwork.py

```python
import numpy as np

def tanh(x):
    return np.tanh(x)

def tan_deriv(x):
    return 1.0 - np.tanh(x) * np.tan(x)

# sigmoid
def logistic(x):
    return 1 / (1 + np.exp(-x))

def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv == logistic_deriv

        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tan_deriv

        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i-1]+1, layers[i]+1))-1)*0.25)
            self.weights.append((2 * np.random.random((layers[i]+1, layers[i+1]))-1)*0.25)


    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)

        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0: -1] = X
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
                deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
```

#####  Run.py

```python
from BPNetwork import NeuralNetwork
import numpy as np

train = [0, 0, 0, 0, 0]
test = [0, 0, 0, 0, 0]
train[0] = [[0,1,1,0,0],
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,1,1,1,0]]

train[1] = [[1,1,1,1,0],
            [0,0,0,0,1],
            [0,1,1,1,0],
            [1,0,0,0,0],
            [1,1,1,1,1]]

train[2] = [[1,1,1,1,0],
            [0,0,0,0,1],
            [0,1,1,1,0],
            [0,0,0,0,1],
            [1,1,1,1,0]]

train[3] = [[0,0,0,1,0],
            [0,0,1,1,0],
            [0,1,0,1,0],
            [1,1,1,1,1],
            [0,0,0,1,0]]

train[4] = [[1,1,1,1,1],
            [1,0,0,0,0],
            [1,1,1,1,0],
            [0,0,0,0,1],
            [1,1,1,1,0]]

test[0] = [[0,0,1,1,0],
           [0,0,1,1,0],
           [0,1,0,1,0],
           [0,0,0,1,0],
           [0,1,1,1,0]]

test[1] = [[1,1,1,1,0],
           [0,0,0,0,1],
           [0,1,1,1,0],
           [1,0,0,0,1],
           [1,1,1,1,1]]

test[2] = [[1,1,1,1,0],
           [0,0,0,0,1],
           [0,1,1,1,0],
           [1,0,0,0,1],
           [1,1,1,1,0]]

test[3] = [[0,1,1,1,0],
           [0,1,0,0,0],
           [0,1,1,1,0],
           [0,0,0,1,0],
           [0,1,1,1,0]]

test[4] = [[0,1,1,1,1],
           [0,1,0,0,0],
           [0,1,1,1,0],
           [0,0,0,1,0],
           [1,1,1,1,0]]

nn = NeuralNetwork([25, 50, 5], 'tanh')
# temp = init[0]
for i in range(5):
    X = np.array(train[i])
    y = np.array(test[i])
    nn.fit(X, y)
    for j in train[i]:
        print(nn.predict(j))

```

### 实验预测测试集结果：

 

 