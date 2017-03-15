""" Data iterator"""
import mxnet as mx
import numpy as np
import sys, os
import cv2
import time
import multiprocessing
import itertools

from scipy import ndimage
from sklearn import neighbors
sys.path.append('../')

from utils import get_rgb_data
from utils import get_spectral_data

from utils import get_polygons
from utils import rasterize_polgygon
from utils import get_raster
from utils import colorize_raster
from utils import get_rgb_image
from utils import unsoft, get_scale_factor, rasterize_polgygon

import tifffile as tiff
import cv2
import numpy as np
import pandas as pd
from shapely import wkt
from shapely import affinity
from rasterio.features import rasterize
from rasterio import features
from shapely import geometry
from collections import defaultdict
from shapely.geometry import MultiPolygon, Polygon
from skimage import measure, exposure

A_data = []
M_data = []
P_data = []
y_mask = []

sf = 24
a_size = 16
m_size = 64
p_size = 128
l_size = 128
n_out = 10

print('sf: {}'.format(sf))


class CropSampler(object):
    ''' Draw a class_i from the class probability distribution;
        Draw a random ImageId with given class_i, from the prev step;
        Sample a crop position from ImageId based on the kde of labels
    '''
    def __init__(self, masks):
        n_class = 10
        self.maps_with_class = [[], [], [], [], [], [], [], [], [], []]
        self.kde_samplers = []
        self.class_probs = np.ones(n_class) / n_class
#        self.class_probs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5])
        self.mask_size = None
        ts = time.time()
        for mask_i, mask in enumerate(masks):
            assert mask.shape[2] == n_class
            if not self.mask_size:
                self.mask_size = mask.shape[1]
            samplers = []
            for class_i in range(n_class):
                X = np.nonzero(mask[:, :, class_i])
                X = np.stack(X, axis=1)

#                np.random.shuffle(X)
#                X = X[:50000]

                if not X.size:
                    samplers.append(None)
                else:
                    self.maps_with_class[class_i].append(mask_i)
                    sampler = neighbors.KernelDensity(self.mask_size * 0.02).fit(X)
                    samplers.append(sampler)

            assert len(samplers) == n_class
            self.kde_samplers.append(samplers)
        print('sampler init time: {}'.format(time.time() - ts))

    def update(self, probs):
        assert self.class_probs.size == probs.size
        self.class_probs = np.copy(probs)

    def sample_crop(self, n):
        kx = np.array([len(x) for x in self.maps_with_class])
        class_hist = np.random.multinomial(n, self.class_probs * (kx != 0))
        class_ids = np.repeat(np.arange(class_hist.shape[0]), class_hist)
        X = []
        for class_id in class_ids:
            for i in range(20):
                random_image_idx = np.random.choice(self.maps_with_class[class_id])
                if random_image_idx < 25:
                    break
            x = self.kde_samplers[random_image_idx][class_id].sample()[0]
            x /= self.mask_size
            x = np.clip(x, 0., 1.)
            return x, class_id, random_image_idx
            X.append(x)
        return X

sampler = None


def flip_mat(mat):
    n_mat = np.zeros(mat.shape, dtype=np.float32)
    for i in range(mat.shape[2]):
        n_mat[:, :, i] = np.fliplr(mat[:, :, i])
    return n_mat


def rot90_mat(mat, k):
    n_mat = np.zeros(mat.shape, dtype=np.float32)
    for i in range(mat.shape[2]):
        n_mat[:, :, i] = np.rot90(mat[:, :, i], k)
    return n_mat


def get_data(image_id, a_size, m_size, p_size, sf):
    rgb_data = get_rgb_data(image_id)
    rgb_data = cv2.resize(rgb_data, (p_size*sf, p_size*sf),
                          interpolation=cv2.INTER_LANCZOS4)

#    rgb_data = rgb_data.astype(np.float) / 2500.
#    print(np.max(rgb_data), np.mean(rgb_data))

#    rgb_data[:, :, 0] = exposure.equalize_adapthist(rgb_data[:, :, 0], clip_limit=0.04)
#    rgb_data[:, :, 1] = exposure.equalize_adapthist(rgb_data[:, :, 1], clip_limit=0.04)
#    rgb_data[:, :, 2] = exposure.equalize_adapthist(rgb_data[:, :, 2], clip_limit=0.04)    

    A_data = get_spectral_data(image_id, a_size*sf, a_size*sf, bands=['A'])
    M_data = get_spectral_data(image_id, m_size*sf, m_size*sf, bands=['M'])
    P_data = get_spectral_data(image_id, p_size*sf, p_size*sf, bands=['P'])

#    lab_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2LAB)
    P_data = np.concatenate([rgb_data, P_data], axis=2)

    return A_data, M_data, P_data


def crop_maps(maps, rel_x, rel_y, rel_size):
    ''' Crop with relative coords
    '''
#    assert all([0. <= rel_x, rel_y, rel_size <= 1.])
    assert rel_x + rel_size <= 1
    res = []
    for m in maps:
        abs_x = int(rel_x * m.shape[1])
        abs_y = int(rel_y * m.shape[1])
        abs_size = int(rel_size * m.shape[1])
        res.append(m[abs_x: abs_x + abs_size, abs_y: abs_y + abs_size])
    return res


def get_crop_position(rel_cx, rel_cy, crop_size, map_size):
    abs_cx = rel_cx * map_size - crop_size / 2.
    abs_cy = rel_cy * map_size - crop_size / 2.
    abs_cx = int(min(max(abs_cx, 0), map_size - crop_size))  # out of border
    abs_cy = int(min(max(abs_cy, 0), map_size - crop_size))
    return abs_cx, abs_cy


def rel_crop(im, rel_cx, rel_cy, crop_size):

    map_size = im.shape[1]
    r = crop_size / 2
    abs_cx = rel_cx * map_size
    abs_cy = rel_cy * map_size
    na = np.floor([abs_cy-r, abs_cy+r, abs_cx-r, abs_cx+r]).astype(np.int32)
    a = np.clip(na, 0, map_size)
    px0 = a[2] - na[2]
    px1 = na[3] - a[3]
    py0 = a[0] - na[0]
    py1 = na[1] - a[1]
    crop = im[a[0]:a[1], a[2]:a[3]]
    crop = np.pad(crop, ((py0, py1), (px0, px1), (0, 0)),
                  mode='reflect')

    assert crop.shape == (crop_size, crop_size, im.shape[2])
    return crop


def get_random_data():
    (y, x), class_id, im_idx = sampler.sample_crop(1)

    a_data_glob = A_data[im_idx]
    m_data_glob = M_data[im_idx]
    p_data_glob = P_data[im_idx]
    label_glob = y_mask[im_idx]

    a_x, a_y = get_crop_position(x, y, a_size, a_data_glob.shape[1])
    m_x, m_y = get_crop_position(x, y, m_size, m_data_glob.shape[1])
    p_x, p_y = get_crop_position(x, y, p_size, p_data_glob.shape[1])
    l_x, l_y = get_crop_position(x, y, l_size, label_glob.shape[1])
    a_data = a_data_glob[a_y: a_y + a_size, a_x: a_x + a_size]
    m_data = m_data_glob[m_y: m_y + m_size, m_x: m_x + m_size]
    p_data = p_data_glob[p_y: p_y + p_size, p_x: p_x + p_size]
    label = label_glob[l_y: l_y + l_size, l_x: l_x + l_size]

#    a_data = rel_crop(a_data_glob, x, y, a_size)
#    m_data = rel_crop(m_data_glob, x, y, m_size)
#    p_data = rel_crop(p_data_glob, x, y, p_size)
#    label = rel_crop(label_glob, x, y, l_size)

#    rgb = colorize_raster(label)
#    cv2.circle(rgb, (int(x * label_glob.shape[1]), int(y * label_glob.shape[1])), 30, (0, 0, 255))
#    cv2.imshow('label', rgb)
#
#    def get_rgb_image1(image, h=None, w=None):
#        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#        for c in range(3):
#            min_val, max_val = np.percentile(image[:, :, c], [2, 98])
#            image[:, :, c] = 255*(image[:, :, c] - min_val) / (max_val - min_val)
#            image[:, :, c] = np.clip(image[:, :, c], 0, 255)
#        image = (image).astype(np.uint8)
#        return image
#
#    rgb_data = get_rgb_image1(p_data[:, :, 0:3])
#    cv2.imshow('rgb_data', rgb_data)
#    cv2.waitKey()

    if np.random.randint(0, 2):
        a_data = flip_mat(a_data)
        m_data = flip_mat(m_data)
        p_data = flip_mat(p_data)
        label = flip_mat(label)

    if np.random.randint(0, 2):
        k = np.random.randint(0, 4)
        a_data = rot90_mat(a_data, k)
        m_data = rot90_mat(m_data, k)
        p_data = rot90_mat(p_data, k)
        label = rot90_mat(label, k)

#    if np.random.randint(0, 2):
#        angle = np.random.randint(0, 180)
#        data = ndimage.interpolation.rotate(data, angle, reshape=False)
#        label = ndimage.interpolation.rotate(label, angle, reshape=False)

#    assert label.shape[:2] == p_data.shape[:2]

    a_data = np.transpose(a_data, (2, 0, 1))
    m_data = np.transpose(m_data, (2, 0, 1))
    p_data = np.transpose(p_data, (2, 0, 1))
    label = np.transpose(label, (2, 0, 1))

    if n_out == 11:
        label = np.argmax(label, axis=0) + (np.max(label, axis=0) != 0)  # 0
        label.shape = (1,) + label.shape

    return a_data, m_data, p_data, label


class Batch(mx.io.DataBatch):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.pad = 0
        self.index = 0

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

#polygons_test = pd.read_csv('blend.csv')
#def get_test_polygons(img_id, h, w):
#    y_sf, x_sf = get_scale_factor(img_id, w, h)
#    polygons = []
#    image = polygons_test[polygons_test.ImageId == img_id]
#    for cType in image.ClassType.unique():
#        wkt_str = image[image.ClassType == cType].MultipolygonWKT.values[0]
#        sh = wkt.loads(wkt_str)
#        sh = affinity.scale(sh, xfact=x_sf, yfact=y_sf, origin=(0, 0, 0))
#        polygons.append(sh)
#    return polygons


class MultiInputSegDataIter(mx.io.DataIter):
    def __init__(self, image_list, batch_size, epoch_size,
                 data_name="data", label_name="softmax_label", start_aug=True, test_list=[]):
        super(MultiInputSegDataIter, self).__init__()
        print('Data iterator initialization..')
        self.data_name = data_name
        self.label_name = label_name
        global y_mask, A_data, M_data, P_data, a_size, p_size, m_size, l_size
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        self.cursor = -1
        self.image_data = []
        self.true_raster = []

        for image_id in image_list:
            a, m, p = get_data(image_id, a_size, m_size, p_size, sf)
            A_data.append(a)
            M_data.append(m)
            P_data.append(p)
            mask = get_raster(image_id, l_size*sf, l_size*sf)
            y_mask.append(mask)

        #  test to train
#        for image_id in test_list:
#            a, m, p = get_data(image_id, a_size, m_size, p_size, sf)
#            A_data.append(a)
#            M_data.append(m)
#            P_data.append(p)
#            polygons = get_test_polygons(image_id, p_size*sf, p_size*sf)
#            y_mask.append(rasterize_polgygon(polygons, p_size*sf, p_size*sf))

        print('number of maps(train + test): {}'.format(len(y_mask)))
        global sampler
        sampler = CropSampler(y_mask)
        print('Sampler is ready.')

        self.a_data_depth = A_data[0].shape[2]
        self.m_data_depth = M_data[0].shape[2]
        self.p_data_depth = P_data[0].shape[2]
        self.label_depth = y_mask[0].shape[2]

        self.thread_number = 4
        self.prefetch_threads = []

        if not start_aug:
            return
        print('Data loaded.')

        self.manager = multiprocessing.Manager()
        self.q = self.manager.Queue(1024)
        for i in range(self.thread_number):
            pt = multiprocessing.Process(target=self.gen, args=[self.q])
            pt.daemon = True
            pt.start()
            self.prefetch_threads.append(pt)
        print('Daemon prefetcher threads started.')

    def gen(self, q):
        while True:
            a, m, p, label = zip(*[get_random_data()
                                 for _ in range(self.batch_size)])
            q.put((a, m, p, label))

    def update_sampler(self, class_weights):
#        print(class_weights)
        class_weights = 1. / (0.02 + class_weights)
#        class_weights /= np.sum(class_weights)
#        class_weights = np.clip(class_weights, 0.1, 0.9)
        class_weights /= np.sum(class_weights)
        sampler.update(class_weights)
#        print(class_weights)

    @property
    def provide_data(self):
        return [('a_data', (self.batch_size, self.a_data_depth,
                            a_size, a_size)),
                ('m_data', (self.batch_size, self.m_data_depth,
                            m_size, m_size)),
                ('p_data', (self.batch_size, self.p_data_depth,
                            p_size, p_size))]

    @property
    def provide_label(self):
        return [('softmax_label', (self.batch_size, 1 if n_out == 11 else 10,
                                   l_size, l_size))]

    def get_batch_size(self):
        return self.batch_size

    def reset(self):
        self.cursor = -1

    def iter_next(self):
        self.cursor += 1
        if(self.cursor < self.epoch_size):
            return True
        else:
            return False

    def next(self):
        if self.iter_next():
            a, m, p, label = self.q.get(True)

            data_all = [mx.nd.array(a), mx.nd.array(m), mx.nd.array(p)]
            label_all = [mx.nd.array(label)]
            data_names = ['a_data', 'm_data', 'p_data']
            label_names = ['softmax_label']

            return Batch(data_names, data_all, label_names, label_all)
        else:
            raise StopIteration

    def close(self):
        for t in self.prefetch_threads:
            t.terminate()
        self.manager.shutdown()

#train_iter = SegDataIter(['6040_2_2'], 8, 128)
#train_iter = mx.io.PrefetchingIter(train_iter)
#
#n_epoch = 100
#ts = time.time()
#for epoch in range(n_epoch):
#    for i, batch in enumerate(train_iter):
#        data = batch.data
#    print('epoch time: {}'.format(time.time() - ts))
#    train_iter.reset()
#    ts = time.time()
#train_iter.close()
