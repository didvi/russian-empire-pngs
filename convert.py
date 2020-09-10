
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage import transform
from skimage import feature
from scipy import ndimage as ndi
import matplotlib.pyplot as plt 

import argparse
import os
import math
import time

# BORDER DETECTION
def detect_borders_canny(img, sigma=3):
    # did not work very well for removing borders
    edges = feature.canny(img, sigma=sigma)
    return edges

def detect_borders_gradient(img, axis=1):
    # calculates horizontal gradient across image in order to detect borders
    # @source https://www.cis.rit.edu/people/faculty/rhody/EdgeDetection.htm#:~:text=Vertical%20edges%20can%20be%20detected,the%20direction%20of%20the%20transition.
    def calc_grad(row):
        before = row[:-2]
        after = row[2:]
        return after - before

    grads = np.apply_along_axis(calc_grad, axis, img)
    b_min = np.min(grads)
    b_max = np.max(grads)
    
    grads = (grads - b_min) / (b_max - b_min) * 255
    return grads > 170

def remove_borders(img, edges):
    """Naively removes borders on left and right side by removing all pixels between image edge and column with
    the largest number of edge detections
    """
    MAX_BORDER_REMOVAL = 100
    # column with largest border detections for the first 10th of the image
    column_sum = np.sum(edges[:, :MAX_BORDER_REMOVAL], axis=0)
    border_i = np.argmax(column_sum)

    # column with largest border detections for the first 10th of the image
    column_sum = np.sum(edges[:, -MAX_BORDER_REMOVAL:], axis=0)
    right_border_i = np.argmax(column_sum) + (img.shape[1] - MAX_BORDER_REMOVAL)
    
    print("Removed borders at index: " + str((border_i, img.shape[1] - right_border_i)))
    return img[:, border_i:right_border_i]

def split_image_into_three(img, naive=True):
    height = img.shape[0]

    if naive:
        # compute the height of each part (just 1/3 of total)
        height = height // 3
        # separate color channels
        b = img[:height]
        g = img[height: 2*height]
        r = img[2*height: 3*height]
        return b, g, r

    else:
        # Note: this does not work at all for most images, since the borders are lightly colored
        # use gradient calculation to determine horizontal borders
        edges = detect_borders_gradient(img, axis=0)
        skio.imshow(edges)
        skio.show()
        # calculate rows with maximum edge detections
        row_sum = np.sum(edges, axis=1)
        split_1 = np.argmax(row_sum[:height // 2])
        split_2 = np.argmax(row_sum[height // 2:])
        print("Splits are " + str((split_1, split_2)))
        skio.imshow(img[:split_1])
        skio.show()
        skio.imshow(img[split_1:split_2])
        skio.show()
        skio.imshow(img[split_2:])
        skio.show()
        return 

# IMAGE SIMILARITY METRICS
def compute_metric(img_1, img_2, metric='ssd'):
    # Computes similarity metric for two images
    if metric == 'gradient':
        def calc_grad(row):
            before = row[:-2]
            after = row[2:]
            return after - before

        grads = np.apply_along_axis(calc_grad, 1, img_1)
        b_min = np.min(grads)
        b_max = np.max(grads)
        img_1 = (grads - b_min) / (b_max - b_min) * 255
        
        grads = np.apply_along_axis(calc_grad, 1, img_2)
        b_min = np.min(grads)
        b_max = np.max(grads)
        img_2 = (grads - b_min) / (b_max - b_min) * 255
        return np.sum((img_1-img_2)**2) / (img_1.shape[0] * img_1.shape[1])
    
    if metric == 'ncc':
        return np.dot(img_1.flatten() / np.linalg.norm(img_1.flatten()), img_2.flatten() / np.linalg.norm(img_2.flatten()))

    if metric == 'ssd':
        return np.sum((img_1-img_2)**2) / (img_1.shape[0] * img_1.shape[1])

# ALIGN IMAGES
def _best_displacement(channel_1, channel_2, min_offset, max_offset, metric):
    min_score = float('inf')
    displacement = ()
    for offset_x in range(min_offset[0], max_offset[0]):
        for offset_y in range(min_offset[1], max_offset[1]):

            channel_2_crop = _apply_displacement(channel_2, (offset_x, offset_y))
            channel_1_crop = _apply_displacement(channel_1, (-offset_x, -offset_y))

            score = compute_metric(channel_1_crop, channel_2_crop, metric)

            if score < min_score:
                min_score = score
                displacement = (offset_x, offset_y)

    return displacement

def _apply_displacement(img, displacement):
    offset_x = displacement[0]
    offset_y = displacement[1]

    # crop image
    crop_img = img[offset_x:] if offset_x >= 0 else img[:offset_x]
    crop_img = crop_img[:, offset_y:] if offset_y >= 0 else crop_img[:, :offset_y]
    return crop_img

def _align_pyramid(channel_1, channel_2, prev_displacement, scale, metric='ssd'):
    # todo test other ending condition
    if scale >= 1:
        return prev_displacement

    # rescale channels
    rescaled_1 = sk.transform.rescale(channel_1, scale)
    rescaled_2 = sk.transform.rescale(channel_2, scale)

    displacement = _best_displacement(rescaled_1, rescaled_2, ((
        prev_displacement[0] - 1) * 2, (prev_displacement[1] - 1) * 2), ((prev_displacement[0] + 1) * 2, (prev_displacement[1] + 1) * 2), metric)

    return _align_pyramid(channel_1, channel_2, displacement, scale * 2, metric=metric)

def align(channel_1, channel_2, method='exhaustive', metric='ssd', max_offset=16):
    # Returns displacement (x, y) of channel 2 where channel 2 is offset to match channel 1 by some metric
    if method == 'pyramid':
        initial_scale = 1/32
        initial_displacement = _best_displacement(sk.transform.rescale(
            channel_1, initial_scale), sk.transform.rescale(channel_2, initial_scale), (-4, -4), (4, 4), metric)
        
        return _align_pyramid(channel_1, channel_2, scale=initial_scale * 2, prev_displacement=initial_displacement)

    if method == 'exhaustive':
        return _best_displacement(channel_1, channel_2, (-max_offset, -max_offset), (max_offset, max_offset), metric)

    return (0, 0)

# MAIN
def main(imname, method, metric, max_offset, border, show):
    tic = time.time()

    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)
    im = sk.img_as_float(im)
    
    # remove borders that would throw off our image metrics
    if not border:
        edges = detect_borders_gradient(im)
        im = remove_borders(im, edges)

    b, g, r = split_image_into_three(im)
    
    # align the images
    displacement_g = align(b, g, method=method,
                           metric=metric, max_offset=max_offset)
    ag = _apply_displacement(g, displacement_g)
    
    # apply displacement for b and r so that the final shape matches
    b = _apply_displacement(b, (-displacement_g[0], -displacement_g[1]))
    r = _apply_displacement(r, (-displacement_g[0], -displacement_g[1]))

    # find displacement for r
    displacement_r = align(ag, r, method=method,
                           metric=metric, max_offset=max_offset)
    ar = _apply_displacement(r, displacement_r)
    
    # apply displacement for b and ag so that the final shape matches
    # assuming that the first part aligned them, displacement by same amount should not
    # modify previous displacements
    b = _apply_displacement(b, (-displacement_r[0], -displacement_r[1]))
    ag = _apply_displacement(ag, (-displacement_r[0], -displacement_r[1]))

    # print displacement
    print("Displacement for G channel: " + str(displacement_g))
    print("Displacement for R channel: " + str(displacement_r))

    # create a color image
    im_out = np.dstack([ar, ag, b])

    # save the image
    fname = "color/color_" + method + "_" + \
        metric + "_" + str(border) + "_" + os.path.basename(imname).split(".")[0] + '.jpg'
    # skio.imsave(fname, im_out)

    toc = time.time()
    print("Time elapsed: " + str(toc - tic))

    # display the image
    if show:
        skio.imshow(im_out)
        skio.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i")
    ap.add_argument("--method", default='exhaustive')
    ap.add_argument("--metric", default='ssd')
    ap.add_argument("--offset", type=int, default=16)
    ap.add_argument("--border", type=bool, default=False)
    ap.add_argument("--show", type=bool, default=True)
    args = ap.parse_args()

    if os.path.isdir(args.i):
        for f in os.listdir(args.i):
            if os.path.isfile(os.path.join(args.i, f)) :
                try:
                    print("Converting " + f)
                    main(os.path.join(args.i, f), args.method, args.metric, args.offset, args.border, args.show)
                except ValueError as e:
                    print(e)
                    continue
    else:
        main(args.i, args.method, args.metric, args.offset, args.border, args.show)
