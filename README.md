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
----
