# MXNet
MXNet的学习之路
----
## 2018-11-9
&emsp;&emsp;完成了第一个线性回归模型的训练。
+ 学习了MXNet自动求梯度过程
	* 首先对需要求梯度的变量调用```attach_grad```函数，申请求梯度过程中要的内存。
	* 然后用```with auto_grad.record()```记录求梯度过程中的信息。
	* 最后用```backward()```函数自动计算梯度。
	* 需要用到梯度时就用```variables.grad```来获取变量的梯度。

+ 学习了从零开始实现线性回归模型网络，训练大致流程如下：
	* 首先取出batch_size的小批量数据
	* 然后用建立的模型net进行前向运算，计算出模型预测结果$y$
	* 在 ```with auto_grad.record()```中计算损失函数的值loss
	* 调用```loss.backward()```计算梯度
	* 用sgd进行迭代优化模型参数
	* 如果未达到目标误差则继续返回第一步训练，否则停止训练

+ 学习了如何生成小批量数据
	* 选定一个batch_size，并计算总体数据量num_examples
	* 用```random.shuffle(num_examples)```随机排序数据
	* 用生成器```yield```每次从总体数据中取出batch_size组数据
	* 代码大体如下
		```python
		def next_minibatch(batch_size,features,labels):

			num_examples = len(features)
			indices = list(range(num_examples))
			random.shuffle(indices)
		with mx.gpu():
			for i in range(0,num_examples,batch_size):
				j = nd.array(indices[i:min(i+batch_size,num_examples)])
				yield features.take(j),labels.take(j)
		```
每天进步一点点~~~~~
-------
## 2018-11-11
&emsp;&emsp;完成了第一个使用gluon接口进行编程的线性回归模型
+ 学习了`mxnet.gluon.data`模块的使用,主要是用来处理输入数据：
	- 使用`mxnet.gluon.data.ArrayDataset`构造一个dataset类型的数据集
	-	可以使用`mxnet.gluon.data.DataLoader`中的方法读取数据，比如shuffle
+ 学习了`gluon.nn`模块,nn模块中定义了大量的神经网络模型,通常的运用过程如下：
	- 使用`net=nn.Sequential()`建立一个神经网络容器，该容器会连接各层神经网络，并将上一层的输出作为下一层的输入
	- 使用`net.add(nn.Dense(1))`来添加一个全连接层，Dense表示全连接层，可以自己定义输出维度
+ 学习了`gluon.loss`模块,其中定义了非常多的损失函数：
	- 使用`loss =gluon.loss.L2Loss()`构建损失函数后，用实例化的loss对输入数据进行计算即可
+ 学习了`gluon.Trainer`模块，其中定义了大量的优化器：
	- 使用`trainer = gluon.Trainer(param)`来定义自己的优化器，具体查看API文档

+ 使用gluon接口进行训练模型的过程如下：
	- 使用`dataset`模型提取出小批量的训练数据
	- 使用`loss`计算损失函数的值，记住需要在`with auto_grad.record()`中进行，记录数值
	- 调用`loss.backward()`计算梯度
	- 使用构建好的优化器`trainer.step(batch_size)`来进行模型参数的迭代优化

其实总体思路上和上一节相似，只不过用了gluon中的一些模块，来构建模型、优化器等等参数
----

## 2018-11-16
&emsp;&emsp; 完成了softmax从零实现、softmax使用gluon接口实现、以及多层感知机从零实现
+ `mxnet.gluon.vision`模块中有大量的数据集，可以使用该模块下载数据集
+	可以使用以下两种方法来将数据加载到gpu中：
	- `mxnet.Context(contexttype= mx.gpu())`
	- `a = a.as_in_context(mx.gpu)`
+  网络较小时，发现使用GPU速度比使用CPU速度还慢，原因是大部分时间用在GPU和CPU之间传送数据，而不是进行网络运算了

**发现一个问题使用 `transforms.ToTensor()`函数,当存储空间是GPU时，会报错，好像是该函数没有在GPU上实现**
**解决方案**：
&emsp;&emsp;可以自己编写函数在读入数据时进行预处理 lambda函数即可，入口参数为 transform

###神经网络的一般搭建步骤
+ 准备小批量数据（构造data_iter类等等）
+ 搭建神经网络模型；
	- 定义参数（使用W.attach_grad()获取求梯度过程的空间)
	- 定义模型运算，组建模型
+ 定义损失函数
+ 定义优化器（sgd、adam等）
+ 训练模型
	- 取出小批量数据
	- 进行前向运算
	- 计算损失函数
	- 进行反向传播
	- 使用优化器对参数进行迭代优化
	- 打印日志、循环训练
------
## 2018-11-17
&emsp;&emsp;完成了L2正则化，以及dropout的实现
+ gluon接口中，`nn.Dense(activation)`选择不同的激活函数
+ 定义 `gluon.Trainer` 时 **'wd'** 关键字是L2正则化的超参数，加入之后网络会加上权重衰减
+ gluon接口中，使用`nn.Dropout(drop_prob)`来添加一个dropout层，其中drop_prob为超参数
+ `nd.random.uniform(low,up)`可以用来生成固定范围的随机数
+ `auto_grad.is_training()`可以用来判断是否在训练
