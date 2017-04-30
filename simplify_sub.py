# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
from shapely import wkt, affinity, ops
import sys
import csv
import multiprocessing
from shapely.geometry import MultiPolygon, Polygon
import numpy as np


def clip_poly(polys):
    new_polys = []
    for poly in polys:
        x, y = poly.exterior.coords.xy
        x = np.clip(x, 0.00001, 0.9999)
        y = np.clip(y, -0.9999, 0.09001)
        new_polys.append(Polygon(shell=zip(x, y)))
    return MultiPolygon(new_polys)

csv.field_size_limit(sys.maxsize)

version = 'v61'
epoch = 2620

shs = []
id_class = []
with open('v65-3950-cv-zero12.csv'.format(version, epoch)) as in_file:
    reader = csv.reader(in_file)
    header = next(reader)
    for i, row in enumerate(reader):
        shs.append((row[0], row[1], row[2]))
#        id_class.append((row[0], row[1]))
        if i % 100 == 99:
            print(i)
#            break
print('csv reading completed')


def func(dat):
    image_id, class_id, sh = dat
    sh = wkt.loads(sh)
#    print(len(sh.wkt))
#    print(len(sh.wkt))
#    if not sh.is_valid:
#    if not isinstance(sh, MultiPolygon):
#        sh = MultiPolygon([sh])
#    sh = clip_poly(sh)
#    if not isinstance(sh, MultiPolygon):
#        sh = MultiPolygon([sh])
#    sh = clip_poly(sh)

    sh = sh.buffer(0.0000000000001)
    sh = sh.simplify(0.00000001, preserve_topology=True)
#    sh = MultiPolygon([x.buffer(0) for x in sh])

#    sh = ops.cascaded_union(sh)
    if not sh.is_valid:
        print(image_id, class_id)
#        qwe
    pol = sh.wkt
#    pol = wkt.dumps(sh, rounding_precision=8)
    return image_id, class_id, pol

pool = multiprocessing.Pool(40)
shs = pool.map(func, shs)
pool.close()
pool.join()
print('simplified')

fo = open('v65-3950-cv-zero12-simp.csv'.format(version, epoch), 'w')
print('ImageId,ClassType,MultipolygonWKT', file=fo)
for image_id, class_id, sh in shs:
    print('{},{},"{}"'.format(image_id, class_id, sh), file=fo)
