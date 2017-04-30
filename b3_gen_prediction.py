from __future__ import print_function
import pandas as pd
import numpy as np
import mxnet as mx
import cv2
import os
from sklearn.externals import joblib
from shapely import affinity
import multiprocessing
from shapely.geometry import MultiPolygon, Polygon
from collections import defaultdict

from utils import polygonize, polygonize_cv, polygonize_sk
from utils import get_scale_factor, root_path, colorize_raster, unsoft

from b3_data_iter import get_data, crop_maps
from b3_data_iter import MultiInputSegDataIter

df = pd.read_csv(root_path + '/sample_submission.csv')
test_list = df.ImageId.unique()
batch_size = 2

version = 'v84'
epoch = 1300
ctx = mx.gpu(0)

n_out = 11

sf = 16
a_size = 16
m_size = 64
p_size = 128
l_size = 128

d = {'a_data': (batch_size, 8, a_size, a_size),
     'm_data': (batch_size, 8, m_size, m_size),
     'p_data': (batch_size, 4, p_size, p_size)
     }


def load_model(version, epoch, batch_size=8, ctx=mx.gpu()):
    sym, arg, aux = mx.model.load_checkpoint('models/' + version, epoch)
    mod = mx.module.Module(sym, context=ctx, data_names=list(d))
    mod.bind(data_shapes=list(d.iteritems()), for_training=False)
    mod.set_params(arg, aux)
    return mod


def predict(d):
    '''get predicted raster
    '''
    image_id, i = d
    mod = load_model(version, epoch, batch_size, mx.gpu(0))

    image_data = get_data(image_id, a_size, m_size, p_size, sf)

    P = []
    M = []
    A = []
    for i in range(sf):
        for j in range(sf):
            rel_size = 1. / sf
            a, m, p = crop_maps(image_data, i*rel_size, j*rel_size, rel_size)
            P.append(p)
            M.append(m)
            A.append(a)

    A = np.array(A).transpose((0, 3, 1, 2))
    M = np.array(M).transpose((0, 3, 1, 2))
    P = np.array(P).transpose((0, 3, 1, 2))
    data_iter = mx.io.NDArrayIter(data={'a_data': A,
                                        'm_data': M,
                                        'p_data': P}, batch_size=batch_size)
    preds = mod.predict(data_iter).asnumpy()

    gg = np.zeros((n_out, l_size*sf, l_size*sf))
    for i in range(sf):  # TODO via reshape
        for j in range(sf):
            gg[:, l_size*i: l_size*(i+1),
               l_size*j: l_size*(j+1)] = preds[i*sf+j]
    preds = gg
#    preds = preds.transpose((1, 2, 0))
    assert preds.shape[0] == n_out
    return preds.astype(np.float32)


def mask_to_poly(dd):
    preds, image_id = dd
    if n_out == 10:
#        preds = (preds > 0.3).astype(np.uint8)
        thresholds = np.array([0.3, 0.3, 0.3, 0.3, 0.4,
                               0.4, 0.3, 0.3, 0.2, 0.2]).reshape((10, 1))
        preds = (preds.reshape((10, -1)) > thresholds).reshape((10, l_size*sf, l_size*sf))
        preds = preds.astype(np.uint8)
    else:
        preds = np.argmax(preds, axis=0)
        preds = unsoft(preds)
        
#    rg = colorize_raster(preds.transpose((1, 2, 0)))

#    cv2.imwrite('1.png', rg)
#    cv2.imshow('mask', rg)
#    cv2.waitKey()

    shs = []
    for i in range(10):
        mask = preds[i]

        y_sf, x_sf = get_scale_factor(image_id, mask.shape[0], mask.shape[1])
        y_sf = 1. / y_sf
        x_sf = 1. / x_sf

        sh = polygonize_cv(mask)
#        sh = polygonize_sk((mask>0)*255, 0)
#        sh = (sh1.buffer(0).intersection(sh2.buffer(0))).buffer(0)

#        if not sh.is_valid:
#            sh = sh.buffer(0)

        sh = affinity.scale(sh, xfact=x_sf, yfact=y_sf, origin=(0, 0, 0))
        shs.append(sh)

    return shs


try:
    os.mkdir('raw_preds/{}-{}'.format(version, epoch))
except:
    pass

#test_list = test_list[220:221]

sip = multiprocessing.Pool(4)
sip_res = sip.imap(predict, zip(test_list, range(len(test_list))), chunksize=1)

mip = multiprocessing.Pool(40)
mip_results = []
for i, mask in enumerate(sip_res):
    joblib.dump(mask, 'raw_preds/{}-{}/{}.pkl'.format(version, epoch, test_list[i]))
    print('sip: {}'.format(i))
    mip_r = mip.map_async(mask_to_poly, [(mask, test_list[i])])
    mip_results.append(mip_r)
print('sip ended')
mip.close()

fo = open('{}-{}-{}.csv'.format(version, epoch, 'cv'), 'w')
print('ImageId,ClassType,MultipolygonWKT', file=fo)

for i, shs in enumerate(mip_results):
    image_id = test_list[i]
    print(image_id)
    shs = shs.get()[0]
    print('mip: {}'.format(i))
    for j, sh in enumerate(shs):
        print('{},{},"{}"'.format(image_id, j + 1, sh.wkt), file=fo)
print('mip ended')
