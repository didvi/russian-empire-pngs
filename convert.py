
import numpy as np
import skimage as sk
import skimage.io as skio
from skimage import transform

import argparse
import os


def compute_metric(img_1, img_2, metric='ssd'):
    # Computes similarity metric for two images
    if metric == 'ncc':
        return np.dot(img_1.flatten() / np.linalg.norm(img_1.flatten()), img_2.flatten() / np.linalg.norm(img_2.flatten()))

    if metric == 'ssd':
        return np.sum((img_1-img_2)**2)

def _best_displacement(channel_1, channel_2, min_offset, max_offset, metric):
    min_score = float('inf')
    displacement = ()
    for offset_x in range(min_offset[0], max_offset[0]):
        for offset_y in range(min_offset[1], max_offset[1]):

            offset_img=np.roll(channel_2, offset_x, axis=1)
            offset_img=np.roll(offset_img, offset_y, axis=0)

            score=compute_metric(channel_1, offset_img, metric)
            if score < min_score:
                min_score=score
                displacement=(offset_x, offset_y)
    return displacement

def _align_pyramid(channel_1, channel_2, prev_displacement, scale, metric='ssd'):
    # todo test other ending condition
    if scale >= 1:
        return prev_displacement

    # rescale channels
    rescaled_1 = sk.transform.rescale(channel_1, scale)
    rescaled_2 = sk.transform.rescale(channel_2, scale)

    displacement = _best_displacement(rescaled_1, rescaled_2, ((prev_displacement[0] - 1) * 2, (prev_displacement[1] - 1) * 2), ((prev_displacement[0] + 1) * 2, (prev_displacement[1] + 1) * 2), metric)

    return _align_pyramid(channel_1, channel_2, displacement, scale * 2, metric=metric)

def align(channel_1, channel_2, method='exhaustive', metric='ssd', max_offset=15):
    # Returns displacement (x, y) of channel 2 where channel 2 is offset to match channel 1 by some metric
    if method == 'pyramid':
        initial_displacement=_best_displacement(sk.transform.rescale(
            channel_1, 1/16), sk.transform.rescale(channel_2, 1/16), (-max_offset, -max_offset), (max_offset, max_offset), metric)
        return _align_pyramid(channel_1, channel_2, scale=1/8, prev_displacement=initial_displacement)

    if method == 'exhaustive':
        return _best_displacement(channel_1, channel_2,(-max_offset, -max_offset), (max_offset, max_offset), metric)

    return (0, 0)


def main(imname, method, metric, max_offset):
    # read in the image
    im=skio.imread(imname)

    # convert to double (might want to do this later on to save memory)
    im=sk.img_as_float(im)

    # compute the height of each part (just 1/3 of total)
    height=np.floor(im.shape[0] / 3.0).astype(np.int)

    # separate color channels
    b=im[:height]
    g=im[height: 2*height]
    r=im[2*height: 3*height]

    # align the images
    displacement_g=align(b, g, method=method,
                         metric=metric, max_offset=max_offset)
    ag=np.roll(g, displacement_g[0], axis=1)
    ag=np.roll(ag, displacement_g[1], axis=0)

    displacement_r=align(b, r, method=method,
                         metric=metric, max_offset=max_offset)
    ar=np.roll(r, displacement_r[0], axis=1)
    ar=np.roll(ar, displacement_r[1], axis=0)

    # print displacement
    print("Displacement for G channel: " + str(displacement_g))
    print("Displacement for R channel: " + str(displacement_r))

    # create a color image
    im_out=np.dstack([ar, ag, b])

    # save the image
    fname="images/color_" + method + "_" + \
        metric + "_" + os.path.basename(imname).split(".")[0] + '.jpg'
    skio.imsave(fname, im_out)

    # display the image
    skio.imshow(im_out)
    skio.show()


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-i")
    ap.add_argument("--method", default='exhaustive')
    ap.add_argument("--metric", default='ssd')
    ap.add_argument("--offset", type=int, default=15)

    args=ap.parse_args()
    main(args.i, args.method, args.metric, args.offset)
