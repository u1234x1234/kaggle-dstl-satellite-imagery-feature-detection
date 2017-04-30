# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import numpy as np
import mxnet as mx
import cv2
import os
import shutil
from sklearn.externals import joblib
from shapely import affinity
import multiprocessing
from shapely.geometry import MultiPolygon, Polygon
from collections import defaultdict

from utils import polygonize, polygonize_cv, polygonize_sk, get_rgb_image
from utils import get_scale_factor, root_path, colorize_raster, unsoft


nets = [
#        ('v57', 1400),
#        ('v58', 1290),
#        ('v59', 3820),
#        ('v60', 3780),
#        ('v61', 2620),
#        ('v62', 1500),
#        ('v63', 2700),
#        ('v74', 390),
#        ('v75', 850)

        ('v65', 3950),
        ('v67', 560),
        ('v68', 1810),
        ('v84', 1300)
#        ('v65', 3950),
#        ('v66', 2300)
        ]

n_out = 11
size = 2048

df = pd.read_csv('input/sample_submission.csv')
test_list = df.ImageId.unique()


def mask_to_poly(image_id):
    global size
    xx = []
    for version, epoch in nets:
        x = joblib.load('/data/raw_preds/{}-{}/{}.pkl'.format(version, epoch, image_id))
        x = cv2.resize(x.transpose((1, 2, 0)), (size, size))
        xx.append(x)

    xx = np.stack(xx, axis=3)
    preds = np.mean(xx, axis=3)
#    preds = joblib.load('raw_preds/{}-{}/{}.pkl'.format('v63', 2700, image_id))
#    preds = cv2.resize(preds.transpose((1, 2, 0)), (size, size))
    preds = preds.transpose((2, 0, 1)).copy()
    print(image_id)

    if n_out == 10:
#        preds = (preds > 0.3).astype(np.uint8)

        thresholds = np.array([0.4, 0.3, 0.3, 0.3, 0.7,
                               0.4, 0.4, 0.4, 0.04, 0.04]).reshape((10, 1))
        preds = (preds.reshape((10, -1)) > thresholds).reshape((10, size, size))
        preds = preds.astype(np.uint8)
    else:
        preds = np.argmax(preds, axis=0)
        preds = unsoft(preds)
        
    rg = colorize_raster(preds.transpose((1, 2, 0)))
    rg_size = 700
    rg = cv2.resize(rg, (rg_size, rg_size))
    im = get_rgb_image(image_id, rg_size, rg_size)
    rg = np.hstack([rg, im])
    cv2.imwrite('raw_blend_temp5/{}.png'.format(image_id), rg)
    joblib.dump(preds.astype(np.float32), 'raw_preds/raw_blend5/{}.pkl'.format(image_id))
    return

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

        try:
            sh = MultiPolygon(sh)
        except:
            print('ERRRROR!!')
            sh = MultiPolygon()

        shs.append(sh)

    return shs

try:
    shutil.rmtree('raw_blend_temp5/')
except:
    pass
os.mkdir('raw_blend_temp5')

#test_list = test_list[:4]
pool = multiprocessing.Pool(16)
res = pool.imap(mask_to_poly, test_list, chunksize=1)
pool.close()
pool.join()
qwe

fo = open('blend_raw5.csv', 'w')
print('ImageId,ClassType,MultipolygonWKT', file=fo)

for i, shs in enumerate(res):
    image_id = test_list[i]
    print('mip: {}'.format(i))
    for j, sh in enumerate(shs):
        print('{},{},"{}"'.format(image_id, j + 1, sh.wkt), file=fo)
print('mip ended')