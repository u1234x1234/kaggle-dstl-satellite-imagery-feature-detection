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

from b3_data_iter import get_data, crop_maps
from b3_data_iter import MultiInputSegDataIter

n_out = 11

df = pd.read_csv('input/sample_submission.csv')
test_list = df.ImageId.unique()


def mask_to_poly(image_id):
    preds = joblib.load('raw_preds/raw_blend5/{}.pkl'.format(image_id))
    size = preds.shape[1]
    if n_out == 10:
#        preds = (preds > 0.3).astype(np.uint8)

        thresholds = np.array([0.4, 0.4, 0.4, 0.4, 0.8,
                               0.4, 0.4, 0.4, 0.1, 0.1]).reshape((10, 1))
        preds = (preds.reshape((10, -1)) > thresholds).reshape((10, size, size))
        preds = preds.astype(np.uint8)
    else:
        preds = np.argmax(preds, axis=0)
        preds = unsoft(preds)
        
    rg = colorize_raster(preds.transpose((1, 2, 0)))
#    cv2.imwrite('1.png', rg)
    size = 900
    rg = cv2.resize(rg, (size, size))
#    cv2.imshow('mask', rg)
#    cv2.waitKey()
    im = get_rgb_image(image_id, size, size)
    rg = np.hstack([rg, im])
    cv2.imwrite('raw_temp5_1/{}.png'.format(image_id), rg)

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
    shutil.rmtree('raw_temp5_1/')
except:
    pass
os.mkdir('raw_temp5_1')

#test_list = test_list[:12]
pool = multiprocessing.Pool(20)
res = pool.imap(mask_to_poly, test_list, chunksize=1)
pool.close()

fo = open('raw_blend5_1.csv', 'w')
print('ImageId,ClassType,MultipolygonWKT', file=fo)

for i, shs in enumerate(res):
    image_id = test_list[i]
    print('mip: {}'.format(i))
    for j, sh in enumerate(shs):
        print('{},{},"{}"'.format(image_id, j + 1, sh.wkt), file=fo)
print('mip ended')
