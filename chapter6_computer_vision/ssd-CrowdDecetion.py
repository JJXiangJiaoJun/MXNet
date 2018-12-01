# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import mxnet as mx
import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import loss as gloss, nn
import time

#输入的num_anchors表示每个像素的锚框数
def cls_pred(num_anchors,num_classes):
    return nn.Conv2D(channels=num_anchors*(num_classes+1),kernel_size=3,padding=1)

def bbox_pred(num_anchors):
    return nn.Conv2D(channels=num_anchors*4,kernel_size=3,padding=1)


def flatten_pred(pred):
    return pred.transpose((0,2,3,1)).flatten()

def concat_preds(preds):
    return nd.concat(*[flatten_pred(pred) for pred in preds],dim=1)

def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.BatchNorm(),nn.Activation('relu'),
               nn.Conv2D(num_channels,kernel_size=3,padding=1))
    #最后接一个最大池化层
    blk.add(nn.MaxPool2D(pool_size=2,strides=2))
    return blk

#可以自己定义比如说resnet等等
def base_net():
    blk = nn.Sequential()
    for num_filters in [16,32,64]:
        blk.add(down_sample_blk(num_filters))
    return blk


def get_blk(i):
    if i==0:
        return base_net()
    elif i==4:
        return nn.GlobalMaxPool2D()
    else:
        return down_sample_blk(128)



def blk_forward(X,blk,sizes,ratios,cls_predictor,bbox_predictor):
    #定义SSD中前向运算的函数
    #生成锚框
    #前向运算
    Y = blk(X)                    #计算下一层的输出 （批量大小，通道数，高，宽）
    anchors = contrib.ndarray.MultiBoxPrior(Y, sizes=sizes, ratios=ratios)
    cls_preds = cls_predictor(Y)   #预测类别 （批量大小，锚框个数*（类别数+1），高，宽）
    bbox_preds = bbox_predictor(Y) #预测边界框回归 （批量大小，锚框个数*4，高，宽）
    
    #进入下一层的运算
    return (Y,anchors,cls_preds,bbox_preds)


class TinySSD(nn.Block):
    def __init__(self,num_classes,**kwargs):
        
        super(TinySSD,self).__init__(**kwargs)
        #定义网络结构
        self.num_classes = num_classes
        #定义每一层的宽高比和锚框个数
        self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
                        [0.88, 0.961]]
        self.ratios = [[1,0.75,0.5,0.35,0.25,0.15,0.05]] * 5
        #定义每一层的的网络结构,特征层+预测类别+预测边界框
        for i in range(5):
            num_anchors_per_pixel = len(self.sizes[i])+len(self.ratios[i])-1
            setattr(self,'blk_%d'%i,get_blk(i))
            setattr(self,'cls_predictor_%d'%i,cls_pred(num_anchors_per_pixel,self.num_classes))
            setattr(self,'bbox_predictor_%d'%i,bbox_pred(num_anchors_per_pixel))
    
    def forward(self,X):
        #定义前向运算，每一个都会输出
        anchors,cls_preds,bbox_preds=[],[],[]
        for i in range(5):
            #前向运算
            X,anchor,cls_pred,bbox_pred = blk_forward(X,getattr(self,'blk_%d' % i),self.sizes[i],self.ratios[i],
                                                     getattr(self,'cls_predictor_%d' %i),getattr(self,'bbox_predictor_%d'%i))
            anchors.append(anchor)
            cls_preds.append(cls_pred)
            bbox_preds.append(bbox_pred)
        
        #返回输出
        return (nd.concat(*anchors,dim=1),
                concat_preds(cls_preds).reshape((0,-1,self.num_classes+1)),
                concat_preds(bbox_preds))

##重新定义损失函数
class focal_SoftMaxCrossEntropyLoss(gloss.Loss):
    def __init__(self,weight=1.0,axis=-1,batch_axis = 0,**kwargs):
        super(focal_SoftMaxCrossEntropyLoss,self).__init__(weight,batch_axis,**kwargs)
        self.axis = axis
        self.weight = weight
    def hybrid_forward(self,F,pred,label,alpha=1.0,gamma=0,weight=1.0):
        #调用真正的SoftMaxCrossEntropyLoss函数
        #首先对最后一维进行softmax运算
        pred_prob = F.softmax(pred)
         #选出相应的类别
        output = nd.pick(pred_prob,label,axis=self.axis)
        #print(output.shape)
        #计算损失函数
        loss = F.mean((-alpha*((1-output)**gamma)*output.log()),axis=self.axis)
        return loss


class smooth_L1Loss(gloss.Loss):
        def __init__(self,weight=1.0,axis=-1,batch_axis=0,**kwargs):
            super(smooth_L1Loss,self).__init__(weight,batch_axis,**kwargs)
            self.axis = axis
            self.weight = weight
        def hybrid_forward(self,F,pred,labels,sigma=0.5):
            #计算每一项差的绝对值
            output = F.abs(pred-labels)
            #计算平滑损失
            smooth_output = F.smooth_l1(output,sigma)
            return F.mean(smooth_output,axis=self.axis)


if __name__ == '__main__':
    #获取人群数据
    batch_size = 64
    edge_size = 256
    param_filename = 'ssd_params'
    train_iter = image.ImageDetIter(path_imgrec='../data/mydataset.rec',
                                    path_imgidx='../data/mydataset.idx',
                                    batch_size=batch_size,
                                    data_shape=(3,edge_size,edge_size),
                                    shuffle= True,
                                    rand_crop=1, #随机裁剪的概率为1
                                    min_object_covered = 0.95,
                                    max_attempts=200)
    ctx = gb.try_gpu()
    ssd = TinySSD(num_classes=1)
    ssd.initialize(init = init.Xavier(),ctx=ctx)
    trainer = gluon.Trainer(ssd.collect_params(),'sgd',{'learning_rate':0.2,'wd':5e-4})
    focal_loss = focal_SoftMaxCrossEntropyLoss()
    smooth_l1 = smooth_L1Loss()

   
    while True:
        #由于不是DataLoader类，每次需要我们重置指针
        train_iter.reset()
        start = time.time()
        train_cls_accuracy = 0
        train_MAE = 0
        train_cls_loss = 0
        #test_iter.reset()
        #每次取出一个小批量数据进行训练
        for i,batch in enumerate(train_iter):
            X = batch.data[0].as_in_context(ctx)
            Y = batch.label[0].as_in_context(ctx)
            #下面进行前向运算
            with autograd.record():
                #计算网络的输出
                anchors,cls_preds,bbox_preds = ssd(X)

                #生成真实标记
                #返回偏移量，掩码，类别标签
                bbox_offsets,bbox_masks,cls_labels=contrib.nd.MultiBoxTarget(anchors,Y,cls_preds.transpose((0,2,1)))
                #下面计算损失函数

                l_cls=focal_loss(cls_preds,cls_labels,0.6,3)
                l_bbox = smooth_l1(bbox_preds*bbox_masks,bbox_offsets*bbox_masks,0.5)
                l = l_cls+l_bbox
                #print(l.shape)
            #反向传播
            l.backward()
            #迭代参数
            trainer.step(batch_size)      
            #记录准确率
            #print(l_cls)
            train_cls_loss += l_cls.mean().asscalar()
            train_MAE += l_bbox.mean().asscalar()
            train_cls_accuracy += eval_cls_accuracy(cls_preds,cls_labels)
            
        print('epoch %2d , class acc %.2f,class err %.2e,bbox mae %.2e,time %.1f sec'
              %(epoch+1,train_cls_accuracy/(i+1),train_cls_loss/(i+1),train_MAE/(i+1),time.time()-start))
        #保存参数
        ssd.save_parameters(param_filename)
            
