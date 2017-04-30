import mxnet as mx
import numpy as np
import sys, os
import cv2
import time
import pandas as pd
from utils import get_rgb_image, get_scale_factor, get_raster, colorize_raster, rasterize_polgygon
from shapely import wkt, affinity
import multiprocessing
from collections import defaultdict
import csv

size = 700
version = 'v84'
epoch = 1300
csv.field_size_limit(sys.maxsize)


df = pd.read_csv('input/sample_submission.csv')
test_list = df.ImageId.unique()

d = defaultdict(list)
with open('{}-{}-cv.csv'.format(version, epoch)) as in_file:
    reader = csv.reader(in_file)
    header = next(reader)
    for qq in range(429):
        rows = [next(reader) for _ in range(10)]
        image_id = rows[0][0]
        for i in range(10):
            d[image_id].append(rows[i][2])
print(len(d))


def f(image_id):
#    if os.path.exists('test_poly_{}_{}/{}.png'.format(version, epoch, image_id)):
#        print(image_id)
#        return
    print('begin: {}'.format(image_id))

    p = d[image_id]
    p = [wkt.loads(x) for x in p]
    y_sf, x_sf = get_scale_factor(image_id, size, size)
    p = [affinity.scale(x, xfact=x_sf, yfact=y_sf, origin=(0, 0, 0)) for x in p]
    rst = rasterize_polgygon(p, size, size)
    color_rst = colorize_raster(rst)
    im = get_rgb_image(image_id, size, size)

    rr = np.hstack([color_rst, im])
    cv2.imwrite('test_poly_{}_{}-cv/{}.png'.format(version, epoch, image_id), rr)
    print('end: {}'.format(image_id))


try:
    os.mkdir('test_poly_{}_{}-cv'.format(version, epoch))
except:
    pass


pool = multiprocessing.Pool(8)
pool.imap(f, test_list, chunksize=1)
pool.close()
pool.join()
