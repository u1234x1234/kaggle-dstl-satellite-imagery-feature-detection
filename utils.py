# -*- coding: utf-8 -*-
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
from skimage import measure

root_path = 'input/'
three_band_path = root_path + '/three_band/'
sixteen_band_path = root_path + '/sixteen_band/'
grid_sizes = pd.read_csv(root_path + '/grid_sizes.csv')
polygons_raw = pd.read_csv(root_path + '/train_wkt_v4.csv')


def get_spectral_data(img_id, h, w, bands=['A', 'M', 'P']):
    res = []
    for waveband in bands:
        image_path = '{}/{}_{}.tif'.format(sixteen_band_path, img_id, waveband)
        image = tiff.imread(image_path)
        if len(image.shape) == 2:  # for panchromatic band
            image.shape = (1,) + image.shape
        image = image.transpose((1, 2, 0))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
        if len(image.shape) == 2:  # for panchromatic band
            image.shape += (1,)
        res.append(image)
    image = np.concatenate(res, axis=2)
    image = image.astype(np.float32)
    return image


def get_rgb_data(img_id):
    image_path = three_band_path + '/' + img_id + '.tif'
    image = tiff.imread(image_path)
    image = image.transpose((1, 2, 0))
    image = image.astype(np.float32)
    return image


def get_rgb_image(img_id, h=None, w=None):
    image = get_rgb_data(img_id)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for c in range(3):
        min_val, max_val = np.percentile(image[:, :, c], [2, 98])
        image[:, :, c] = 255*(image[:, :, c] - min_val) / (max_val - min_val)
        image[:, :, c] = np.clip(image[:, :, c], 0, 255)
    image = (image).astype(np.uint8)
    if h and w:
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def get_polygons(img_id, h, w):
    y_sf, x_sf = get_scale_factor(img_id, w, h)
    polygons = []
    image = polygons_raw[polygons_raw.ImageId == img_id]
    for cType in image.ClassType.unique():
        wkt_str = image[image.ClassType == cType].MultipolygonWKT.values[0]
        sh = wkt.loads(wkt_str)
        sh = affinity.scale(sh, xfact=x_sf, yfact=y_sf, origin=(0, 0, 0))
        polygons.append(sh)
    return polygons


def get_polygon_train_image(img_id, h, w):
    im = cv2.imread('train_poly/{}.png'.format(img_id))
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return im


def get_grid_size(img_id):
    i_grid_size = grid_sizes[grid_sizes.iloc[:, 0] == img_id]
    x_max = i_grid_size.Xmax.values[0]
    y_min = i_grid_size.Ymin.values[0]
    return y_min, x_max


def get_scale_factor(img_id, h, w):
    w = float(w)
    h = float(h)
    y_min, x_max = get_grid_size(img_id)
    w_ = w * (w/(w+1))
    h_ = h * (h/(h+1))
    x_sf = w_ / x_max
    y_sf = h_ / y_min
    return y_sf, x_sf


def get_raster(img_id, h, w):
    polygons = get_polygons(img_id, h, w)
    return rasterize_polgygon(polygons, h, w)


def rasterize_polgygon(polygons, h, w):
    r = []
    for pol in polygons:
        result = rasterize([pol], out_shape=(h, w))
        r.append(result)
    r = np.stack(r)
    r = r.transpose((1, 2, 0))
    return r


def colorize_raster(masks):
    ''' (H, W, 10) -> (H, W, 3)
    '''
    assert masks.shape[2] == 10
    palette = np.array([(180, 180, 180), (100, 100, 100),  # Buildings, Misc.
                        (6, 88, 179), (125, 194, 223),  # Road, Track
                        (55, 120, 27), (160, 219, 166),  # Trees, Crops
                        (209, 173, 116), (180, 117, 69),  # Waterway, Standing
                        (67, 109, 244), (39, 48, 215)], dtype=np.uint8)  # Car

    r = []
    for obj_type in range(10):
        c = palette[obj_type]
        result = np.stack([masks[:, :, obj_type]] * 3, axis=2)
        r.append(result * c)
    r = np.stack(r)
    r = np.max(r, axis=0)
    return r


def polygonize(im):
    assert len(im.shape) == 2
    shapes = features.shapes(im)
    polygons = []
    for i, shape in enumerate(shapes):
#        if i % 10000 == 0:
#            print(i)
        if shape[1] == 0:
            continue
        polygons.append(geometry.shape(shape[0]))
#    polygons = [geometry.shape(shape[0]) for shape in shapes if shape[1] > 0]
    mp = geometry.MultiPolygon(polygons)
    return mp


def jaccard_raster(true_raster, pred_raster):
    assert true_raster.shape[2] == pred_raster.shape[2] == 10
    score = []
    for i in range(10):
        true = true_raster[:, :, i] != 0
        pred = pred_raster[:, :, i] != 0
        tp = np.sum(true * pred)
        fp = np.sum(pred) - tp
        fn = np.sum(true) - tp
        if tp == 0:
            jac = 0.
        else:
            jac = tp / float(fp + fn + tp)
        score.append((tp, fp, fn, jac))
    score = np.array(score)
    assert score.shape == (10, 4)
    return score


def unsoft(max_idx, num_class=11):
    max_idx = max_idx.squeeze()
    assert len(max_idx.shape) == 2
    preds = np.zeros((num_class - 1, max_idx.shape[0], max_idx.shape[1]))
    for i in range(1, num_class):
        preds[i - 1] = (max_idx == i)
    preds = preds.astype(np.uint8)
    return preds


def polygonize_cv(mask, epsilon=1., min_area=10.):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    approx_contours = contours
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def polygonize_sk(mask, level):
    contours = measure.find_contours(mask, level)
    polys = []
    for contour in contours:
        if contour.shape[0] < 4:
            continue
        poly = Polygon(shell=contour[:, [1, 0]])
        polys.append(poly)
    polys = MultiPolygon(polys)
    return polys
