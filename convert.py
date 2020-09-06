
import numpy as np
import skimage as sk
import skimage.io as skio
import argparse
import os

def compute_metric(img_1, img_2, metric='ssd'):
    # Computes similarity metric for two images
    if metric == 'ssd':
        return np.sum((img_1-img_2)**2)
    else:
        return np.sum((img_1-img_2)**2)

def align(channel_1, channel_2, method='exhaustive', metric='ssd'):
    # Returns displacement (x, y) of channel 2 where channel 2 is offset to match channel 1 by some metric   
    if method == 'exhaustive':
        min_score = float('inf')
        displacement = ()
        
        for offset_x in range(-15, 15):
            for offset_y in range(-15, 15):
                
                offset_img = np.roll(channel_2, offset_x, axis=1)
                offset_img = np.roll(offset_img, offset_y, axis=0)
                
                score = compute_metric(channel_1, offset_img, metric)
                if score < min_score:
                    min_score = score
                    displacement = (offset_x, offset_y)

    return displacement

def main(imname, method, metric):
    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # align the images
    displacement_g = align(b, g, method=method, metric=metric)
    ag = np.roll(g, displacement_g[0], axis=1)
    ag = np.roll(ag, displacement_g[1], axis=0)
    
    displacement_r = align(b, r, method=method, metric=metric)
    ar = np.roll(r, displacement_r[0], axis=1)
    ar = np.roll(ar, displacement_r[1], axis=0)
    
    # print displacement
    print("Displacement for G channel: " + str(displacement_g))
    print("Displacement for R channel: " + str(displacement_r))
    
    # create a color image
    im_out = np.dstack([ar, ag, b])

    # save the image
    fname = "images/color_" + method + "_" + metric + "_" + os.path.basename(imname)
    skio.imsave(fname, im_out)

    # display the image
    skio.imshow(im_out)
    skio.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i")
    ap.add_argument("--method")
    ap.add_argument("--metric")

    args = ap.parse_args()
    main(args.i, args.method, args.metric)