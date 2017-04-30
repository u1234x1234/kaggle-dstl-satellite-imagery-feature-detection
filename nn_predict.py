from __future__ import print_function
import pandas as pd
import numpy as np
import mxnet as mx
import cv2
from utils import root_path
from utils import unsoft
from utils import colorize_raster
from data_iter import get_image_data
from utils import polygonize
from shapely import affinity
from utils import get_scale_factor

df = pd.read_csv(root_path + '/sample_submission.csv')
test_list = df.ImageId.unique()

#test_list = test_list[:1]
batch_size = 2
g_size = 720
l_size = 180
assert g_size % l_size == 0
h = w = l_size
batch_size = 2

fo = open('v6-3.csv', 'w')
print('ImageId,ClassType,MultipolygonWKT', file=fo)


def predict(mod, image_id, g_size, l_size):
    n = g_size / l_size

    image_data = get_image_data(image_id, g_size, g_size)

    patch_data = []
    for i in range(n):
        for j in range(n):
            patch_data.append(image_data[l_size*i: l_size*(i+1),
                                         l_size*j: l_size*(j+1)])
    patch_data = np.array(patch_data)
    patch_data = patch_data.transpose((0, 3, 1, 2))

    data_iter = mx.io.NDArrayIter(data=patch_data, batch_size=batch_size)
    preds = mod.predict(data_iter).asnumpy()
    gg = np.zeros((11, g_size, g_size))
    for i in range(n):  # TODO via reshape
        for j in range(n):
            gg[:, l_size*i: l_size*(i+1),
               l_size*j: l_size*(j+1)] = preds[i*n+j]
    preds = gg

    preds = np.argmax(preds, axis=0)
    preds = unsoft(preds)
#    pred_raster = preds.transpose((1, 2, 0))

    assert preds.shape[0] == 10
    y_sf, x_sf = get_scale_factor(image_id, g_size, g_size)
    y_sf = 1. / y_sf
    x_sf = 1. / x_sf
    for i in range(10):
        sh = polygonize(preds[i])
        sh = affinity.scale(sh, xfact=x_sf, yfact=y_sf, origin=(0, 0, 0))
        print('{},{},"{}"'.format(image_id, i + 1, sh.wkt), file=fo)

        
        
#    y_pred = colorize_raster(pred_raster)
#    cv2.imshow('full pred', y_pred)
#    cv2.waitKey()

sym, arg, aux = mx.model.load_checkpoint('models/v6', 324)
mod = mx.module.Module(sym, context=mx.gpu(0))
mod.bind(data_shapes=[('data', (batch_size, 20, l_size, l_size))],
         for_training=False)
mod.set_params(arg, aux)

print('model loaded!')

for i, image_id in enumerate(test_list):
    predict(mod, image_id, g_size, l_size)
    print(i)
