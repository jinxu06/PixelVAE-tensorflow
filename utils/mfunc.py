import numpy as np
import math
from utils.mask import *

def mask_inputs(inputs, mgen):
    batch_size, height, width, num_channel = inputs.shape
    masks = mgen.gen(batch_size)
    inputs = np.concatenate([inputs, np.expand_dims(masks, axis=-1)], axis=-1)
    for c in range(num_channel):
        inputs[:, :, :, c] *= masks
    return inputs

def find_contour(mask):
    contour = np.zeros_like(mask)
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                lower_bound = max(y-1, 0)
                upper_bound = min(y+1, h-1)
                left_bound = max(x-1, 0)
                right_bound = min(x+1, w-1)
                nb = mask[lower_bound:upper_bound+1, left_bound:right_bound+1]
                if np.min(nb)  == 0:
                    contour[y, x] = 1
    return contour

def tile_images(imgs, size=(6, 6)):
    imgs = imgs[:size[0]*size[1], :, :, :]
    img_h, img_w = imgs.shape[1], imgs.shape[2]
    all_images = np.zeros((img_h*size[0], img_w*size[1], 3), np.uint8)
    for j in range(size[0]):
        for i in range(size[1]):
            all_images[img_h*j:img_h*(j+1), img_w*i:img_w*(i+1), :] = imgs[j*size[1]+i, :, :, :]
    return all_images


def random_crop_images(inputs, output_size):
    bsize, input_h, input_w = inputs.shape[:3]
    output_h, output_w = output_size
    if input_h==output_h and input_w==output_w:
        return inputs, None
    x = []
    y = []
    for i in range(bsize):
        coor_h = np.random.randint(low=0, high=input_h-output_h+1)
        coor_w = np.random.randint(low=0, high=input_w-output_w+1)
        x.append(inputs[i][coor_h:coor_h+output_h, coor_w:coor_w+output_w, :])
        #y.append([float(coor_h)/(input_h-output_h), float(coor_w)/(input_w-output_w)])
        y.append([coor_h, coor_h+output_h, coor_w, coor_w+output_w])
    x = np.array(x)
    y = np.array(y)
    return x, y




def tile_crop_images(input, output_size):
    input_h, input_w, _ = input.shape
    output_h, output_w = output_size
    bsize = (input_h // output_h) * (input_w // output_w)
    inputs = [input.copy() for i in range(bsize)]

    x = []
    y = []
    coor_h = 0
    coor_w = 0
    for i in range(bsize):
        h = i // (input_w // output_w)
        w = i - h * (input_w // output_w)
        coor_h = h * output_h
        coor_w = w * output_w
        x.append(inputs[i][coor_h:coor_h+output_h, coor_w:coor_w+output_w, :])
        y.append([float(coor_h)/(input_h-output_h), float(coor_w)/(input_w-output_w)])
    x = np.array(x)
    y = np.array(y)
    return x, y


def find_next_missing_pixel(mask):
    height, width = mask.shape
    for h in range(height):
        for w in range(width):
            if mask[h, w] == 0:
                return (h, w)

def find_maximally_conditioned_window(mask, window_size, pixel):
    height, width = mask.shape
    h, w = pixel
    # h0, w0 = h-window_size+1, w-window_size//2
    h0, w0 = h-window_size//2, w-window_size//2
    h0, w0 = min(max(h0, 0), height-window_size), min(max(w0, 0), width-window_size)
    return [[h0, h0+window_size], [w0, w0+window_size]]

def broadcast_mask(mask, num_channel=None, batch_size=None):
    if num_channel is not None:
        mask = np.stack([mask.copy() for i in range(num_channel)], axis=-1)
    if batch_size is not None:
        mask = np.stack([mask.copy() for i in range(batch_size)], axis=0)
    return mask




## https://github.com/aizvorski/video-quality/blob/master/psnr.py

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def batch_psnr(imgs1, imgs2, output_mean=True):
    assert imgs1.shape[0]==imgs2.shape[0], "batch size of imgs1 and imgs2 should be the same"
    batch_size = imgs1.shape[0]
    v = [psnr(imgs1[i], imgs2[i]) for i in range(batch_size)]
    if output_mean:
        return np.mean(v)
    return v


def evaluate(original_imgs, completed_imgs):
    return batch_psnr(original_imgs, completed_imgs)
