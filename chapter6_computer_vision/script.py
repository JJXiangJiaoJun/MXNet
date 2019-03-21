import mxnet as mx
import sys
sys.path.insert(0, '..')


import gluonbook as gb
from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import loss as gloss, nn
import time
import os


def load_data_pikachu(batch_size,edge_size=256):
    data_dir ='../data/pikachu'

    train_iter = image.ImageDetIter(path_imgrec=os.path.join(data_dir,'train.rec'),
                                    path_imgidx=os.path.join(data_dir,'train.idx'),
                                    batch_size=batch_size,
                                    data_shape  = (3,edge_size,edge_size),#输出图像形状
                                    shuffle = True,
                                    rand_crop=1, #随机裁剪的概率为1
                                    min_object_covered = 0.95,max_attempts=200,
                                    )
    val_iter = image.ImageDetIter(
                                  path_imgrec=os.path.join(data_dir, 'val.rec'), batch_size=batch_size,
                                 data_shape=(3, edge_size, edge_size), shuffle=False)
    return train_iter,val_iter
gloss.L1Loss

#获取皮卡丘数据
batch_size = 64
train_iter,test_iter = gb.load_data_pikachu(batch_size)


try:
    for i,batch in enumerate(train_iter):
        print(i)
except:
    print('end')