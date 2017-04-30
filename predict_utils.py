# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
from utils import unsoft
from utils import jaccard_raster


def load_model(version, epoch, patch_size, batch_size=8, ctx=mx.gpu()):
    sym, arg, aux = mx.model.load_checkpoint('models/' + version, epoch)
    mod = mx.module.Module(sym, context=ctx)
    mod.bind(data_shapes=[('data', (batch_size, 20, patch_size, patch_size))],
             for_training=False)
    mod.set_params(arg, aux)
    return mod


def predict(mod, image_data, patch_size, map_size):
    n = map_size / patch_size

    patch_data = []
    for i in range(n):
        for j in range(n):
            patch_data.append(image_data[patch_size*i: patch_size*(i+1),
                                         patch_size*j: patch_size*(j+1)])
    patch_data = np.array(patch_data)
    patch_data = patch_data.transpose((0, 3, 1, 2))

    data_iter = mx.io.NDArrayIter(data=patch_data, batch_size=8)
    preds = mod.predict(data_iter).asnumpy()

    gg = np.zeros((10, map_size, map_size))

    for i in range(n):  # TODO via reshape
        for j in range(n):
            gg[:, patch_size*i: patch_size*(i+1),
               patch_size*j: patch_size*(j+1)] = preds[i*n+j]
    preds = gg
    return preds


def calc_jaccard(mod, X_data, y_mask, patch_size, map_size):
    tp, fp, fn = [], [], []
    assert len(X_data) == len(y_mask)
    for i in range(len(X_data)):
        preds = predict(mod, X_data[i], patch_size, map_size)
#        preds = np.argmax(preds, axis=0)
#        preds = unsoft(preds)

        thresholds = np.array([0.4, 0.4, 0.4, 0.4, 0.4,
                               0.4, 0.4, 0.4, 0.4, 0.4]).reshape((10, 1))
#        for i in range(len(thresholds)):
#            preds[i] = (preds[i] > thresholds[i])
        preds = (preds.reshape((10, -1)) > thresholds).reshape((10, map_size, map_size))
        preds = preds.astype(np.uint8)

        pred_raster = preds.transpose((1, 2, 0))

        score = jaccard_raster(y_mask[i], pred_raster)
        tp.append(score[:, 0])
        fp.append(score[:, 1])
        fn.append(score[:, 2])

    tp = np.stack(tp)
    fp = np.stack(fp)
    fn = np.stack(fn)
    tp = np.mean(tp, axis=0)
    fp = np.mean(fp, axis=0)
    fn = np.mean(fn, axis=0)
    jac = tp / (tp + fp + fn)
    jac[np.isnan(jac)] = 0
    return jac
