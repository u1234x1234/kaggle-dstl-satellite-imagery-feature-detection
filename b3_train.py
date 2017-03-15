# -*- coding: utf-8 -*-
import mxnet as mx
import cv2
import pandas as pd
import time
import multiprocessing
import sys
import logging
import numpy as np
import datetime
sys.path.append('../')

from utils import colorize_raster
from utils import get_raster
from utils import jaccard_raster

from b3_data_iter import unsoft
from b3_data_iter import get_data
from b3_data_iter import a_size, m_size, p_size
from b3_data_iter import MultiInputSegDataIter


np.set_printoptions(2, suppress=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

batch_size = 8
ctx = [mx.gpu(0)]
version = 'v96'
print(version)

d = {
     'a_data': (batch_size, 8, a_size, a_size),
     'm_data': (batch_size, 8, m_size, m_size),
     'p_data': (batch_size, 4, p_size, p_size)
     }

def print_inferred_shape(net):
    ar, ou, au = net.infer_shape(**d)
    print(net.name, ou)

def up_module(net, num_filter, name=''):
    net = mx.sym.Deconvolution(net, kernel=(4, 4), pad=(1, 1), stride=(2, 2),
                               num_filter=num_filter, name=name+'_deconv', no_bias=True)
#    net = mx.sym.UpSampling(net, scale=2, sample_type='nearest')
#    net = mx.symbol.Convolution(data=net, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=num_filter)
    
    net = mx.sym.BatchNorm(net, name=name+'_bn')
#    net = mx.sym.InstanceNorm(net)

    net = mx.sym.LeakyReLU(net, act_type='leaky', name=name+'_act')
#    net = mx.sym.Activation(net, act_type='relu')
    return net


def down_module(net, kernel_size, pad_size, num_filter, stride=(1, 1), dilate_size=(1, 1), name='', down=False, dilate=False):
#    if down:
#        stride=(2, 2)

    if dilate:
        dilate_size = (3, 3)
        pad_size = (3, 3)
    
    net = mx.sym.Convolution(data=net, kernel=kernel_size, stride=stride, dilate=dilate_size,
                             pad=pad_size, num_filter=num_filter, name='{}_conv'.format(name), no_bias=True)


    net = mx.sym.BatchNorm(net, name=name+'_bn')
#    net = mx.sym.InstanceNorm(net)

#    net = mx.sym.Activation(net, act_type='relu')
    net = mx.sym.LeakyReLU(net, act_type='leaky', name=name+'_act')

    if down:
        net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2), name=name+'_pool')

    return net


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, num_group=32, bn_mom=0.9, workspace=256, memonger=False):
    if bottle_neck:        
        conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter*0.5), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
#        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        bn1 = mx.sym.InstanceNorm(conv1)
        
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        
        conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.5), num_group=num_group, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
#        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        bn2 = mx.sym.InstanceNorm(conv2)
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')

        
        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
#        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        bn3 = mx.sym.InstanceNorm(conv3)
        
        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
#            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_sc_bn')
            shortcut = mx.sym.InstanceNorm(shortcut_conv)
            
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        eltwise =  bn3 + shortcut
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')


def conv_3d(net, num_filter, name='3d'):
    net = mx.sym.Reshape(net, shape=(0, -4, 1, -1, -2))
    net = mx.sym.Convolution(data=net, kernel=(64, 3, 3), stride=(1, 1, 1), pad=(1,1,1), num_filter=10, name='{}_conv'.format(name))
    net = mx.sym.Reshape(net, shape=(0, -1, 128, 128))

    net = mx.sym.Convolution(data=net, kernel=(3, 3), num_filter=10, name='{}_conv'.format(name))

    net = mx.sym.InstanceNorm(net)
    net = mx.sym.LeakyReLU(net, act_type='leaky', name=name+'_act')
    return net


def get_symbol():
    kernel_size = (3, 3)
    pad_size = (1, 1)
    num_filter = 64

    a_data = mx.sym.Variable("a_data")
    a_data = mx.sym.BatchNorm(a_data)
#    a_data = mx.sym.InstanceNorm(a_data)

    m_data = mx.sym.Variable("m_data")
    m_data = mx.sym.BatchNorm(m_data)
#    m_data = mx.sym.InstanceNorm(m_data)

    p_data = mx.sym.Variable("p_data")
    p_data = mx.sym.BatchNorm(p_data)
#    p_data = mx.sym.InstanceNorm(p_data)


    ######## p down
#    p_net = p_data
    p_net = down1 = down_module(p_data, kernel_size, pad_size, num_filter=num_filter, name='p_down1', down=True)
    p_net = down2 = down_module(p_net, kernel_size, pad_size, num_filter=num_filter*2, name='p_down2', down=True)
#    p_net = down3 = down_module(p_net, kernel_size, pad_size, num_filter=num_filter*3, name='p_down3', down=True)
    
    ######## m down
    m_net = m_data
    m_net = down4 = down_module(m_data, kernel_size, pad_size, num_filter=num_filter, name='m_down1', down=True)

    ######## a down
    a_net = a_data
    a_net = down_module(a_data, kernel_size, pad_size, num_filter=num_filter, name='a_down1', down=True)

    ######## p, m concat
    pm_net = mx.sym.Concat(*[p_net, m_net])
    pm_net = pm_net_down1 = down_module(pm_net, kernel_size, pad_size, num_filter=num_filter*3, name='pm_down1', down=True)
    pm_net = down_module(pm_net, kernel_size, pad_size, num_filter=num_filter*4, name='pm_down2', down=True)

    ######## pm, a concat
    pma_net = mx.sym.Concat(*[pm_net, a_net])
    pma_net = down_module(pma_net, kernel_size, pad_size, num_filter=num_filter*4, name='pma_down1', down=False)
#    pma_net = down_module(pma_net, kernel_size, pad_size, num_filter=num_filter*3, name='pma_down2', down=True)
#    pma_net = down_module(pma_net, kernel_size, pad_size, num_filter=num_filter*4, name='pma_down3', down=True)

    net = pma_net

#    net = down_module(net, kernel_size, pad_size, num_filter=num_filter*4, name='pma_down1', down=True, dilate=False)
#    net = down_module(net, kernel_size, pad_size, num_filter=num_filter*2, name='convup2', down=False, dilate=False)

    net = up_module(net, num_filter=num_filter*4, name='up1')
#    print_inferred_shape(net)
#    qwe

#    net = mx.sym.Concat(*[net, pm_net_down1])
#    net = up_module(net, num_filter=num_filter*3, name='up2')

    net = down_module(net, kernel_size, pad_size, num_filter=num_filter*3, name='convup2', down=False, dilate=False)
    net = up_module(net, num_filter=num_filter*3, name='up2')

#    net = mx.sym.Concat(*[net, m_net, down2])
    net = down_module(net, kernel_size, pad_size, num_filter=num_filter*2, name='convup3', down=False, dilate=False)
    net = up_module(net, num_filter=num_filter*2, name='up3')

    net = mx.sym.Concat(*[net, down1])
    net = down_module(net, kernel_size, pad_size, num_filter=num_filter, name='convup4', down=False, dilate=False)
    net = up_module(net, num_filter=num_filter, name='up4')


#    p_ref = down_module(p_data, kernel_size, pad_size, num_filter=num_filter, name='ref', down=False, dilate=False)
#    net = mx.sym.Concat(*[net, p_ref])
#    net = down_module(net, kernel_size, pad_size, num_filter=num_filter, name='convup5', down=False, dilate=False)

#    net = up_module(net, num_filter=num_filter, name='up5')

#    net = residual_unit(net, num_filter=num_filter*2, stride=(1, 1), dim_match=True, name='res', num_group=1)


#    net = mx.sym.Concat(*[net, down1])
#    net = down_module(net, kernel_size, pad_size, num_filter=num_filter*2, name='convup4', down=False)

#    net = up_module(net, num_filter=num_filter*2, name='up4')

#    net = up_module(net, num_filter=num_filter, name='up5')
    
#    net = down_module(net, kernel_size, pad_size, num_filter=num_filter*2, name='convup5', down=False)
#    net = up_module(net, num_filter=num_filter, name='up5')

#    x = down_module(p_data, kernel_size, pad_size, num_filter=num_filter, name='convup_p', down=False)

#    net = down_module(net, kernel_size, pad_size, num_filter=num_filter, name='convup6', down=False)
#    net = up_module(net, num_filter=num_filter, name='up5')

    net = mx.sym.Convolution(data=net, kernel=(5,5), stride=(1,1),
                             pad=(2,2), num_filter=10, name='final', no_bias=True)


#    print_inferred_shape(net)
#    net = conv_3d(net, 10)
#    print_inferred_shape(net)
#    qwe

#    net = mx.symbol.SoftmaxOutput(data=net, multi_output=True, use_ignore=False, ignore_label=0, name="softmax")
#    net = mx.symbol.LinearRegressionOutput(net, name='softmax')
    net = mx.symbol.LogisticRegressionOutput(net, name='softmax')
#    print_inferred_shape(net)
    return net


sym = get_symbol()
sym.list_arguments()

all_list = pd.read_csv('input/train_wkt_v4.csv').ImageId.unique().tolist()

test_list = ['6040_1_0', '6060_2_3', '6070_2_3', '6120_2_2', '6170_2_4']  # v3
#all_list = ['6070_2_3', '6100_1_3', '6110_3_1', '6010_4_2']
#all_list = ['6100_2_2']
#test_list = pd.read_csv('blend.csv').ImageId.unique().tolist()
#test_list = [x for x in test_list if not x.startswith('6080')]
#train_list += test_list
#train_list = list(set(all_list) - set(test_list))
#all_list = train_list + test_list
train_list = all_list

print('Number image in train: {}'.format(len(train_list)))
print('Number image in test: {}'.format(len(test_list)))

train_iter = MultiInputSegDataIter(train_list, batch_size, 500, test_list=[])
#train_iter = mx.io.PrefetchingIter(train_iter)

print('Data shapes: ', train_iter.provide_data, train_iter.provide_label)

a = mx.viz.plot_network(sym, shape=d)
a.render('{}.pdf'.format(version))

#args, out, aux = sym.infer_shape(data=train_iter.provide_data[0][1])
#for n, s in zip(sym.list_arguments(), args):
#    print(n, s)
#print('-'*30)
#for n, s in zip(sym.list_outputs(), out):
#    print(n, s)
#print('-'*30)
#internals = sym.get_internals()
#_, out, _ = internals.infer_shape(data=train_iter.provide_data[0][1])
#for n, s in zip(internals.list_outputs(), out):
#    print(n, s)
#print('-'*30)

mod = mx.module.Module(sym, context=ctx, data_names=['a_data', 'm_data', 'p_data'])
mod.bind(data_shapes=train_iter.provide_data,
         label_shapes=train_iter.provide_label)
mod.init_params(initializer=mx.initializer.Xavier())
mod.init_optimizer(optimizer=mx.optimizer.Adam())


class SegMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(SegMetric, self).__init__('seg-metric')
        self.cnt = 0
        self.tp = []
        self.fp = []
        self.fn = []

    def update(self, labels, preds):
        preds = preds[0][0].asnumpy()
        labels = labels[0][0].asnumpy()

        if preds.shape[0] == 11:
            preds = np.argmax(preds, axis=0)
            preds = unsoft(preds)
            labels = unsoft(labels)
        else:
            preds = preds > 0.5

        labels = labels.transpose((1, 2, 0))
        preds = preds.transpose((1, 2, 0))
        score = jaccard_raster(labels, preds)
        self.tp.append(score[:, 0])
        self.fp.append(score[:, 1])
        self.fn.append(score[:, 2])
        return

    def aggregate_jaccard_score(self):
        tp = np.stack(self.tp)
        fp = np.stack(self.fp)
        fn = np.stack(self.fn)
        tp = np.mean(tp, axis=0)
        fp = np.mean(fp, axis=0)
        fn = np.mean(fn, axis=0)
        jac = tp / (tp + fp + fn)
        jac[np.isnan(jac)] = 0
        for i in range(10):
            print('Patch-based jaccard {}: {:.2f}'.format(i, jac[i]))
        print('Patch-based jaccard: {}'.format(np.mean(jac)))
        return jac

n_epoch = 5000
for epoch in range(1, n_epoch):
    ts = time.time()
    acc = mx.metric.Accuracy()
#    mse = mx.metric.RMSE()
    sgm = SegMetric()

    for i, data_batch in enumerate(train_iter):
        mod.forward_backward(data_batch)
        mod.update()
#        mod.update_metric(eval_metric=mse, labels=data_batch.label)
        mod.update_metric(eval_metric=sgm, labels=data_batch.label)

#    print('train mse: ', epoch, mse.sum_metric / mse.num_inst)
    jac_acc = sgm.aggregate_jaccard_score()
    train_iter.update_sampler(jac_acc)

    if epoch % 10 == 0:
        mx.model.save_checkpoint('models/' + version, epoch,
                                 mod.symbol, *mod.get_params())
    print('epoch: {} time: {}'.format(epoch, time.time() - ts))
    print(datetime.datetime.now())
    train_iter.reset()
