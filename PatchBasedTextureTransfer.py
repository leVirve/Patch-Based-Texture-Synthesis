#!/usr/bin/python
import cv2
import sys
from random import randint

import numpy as np
from sklearn.feature_extraction import image


# texure = cv2.imread('tex.jpg')
# out = img = cv2.imread('meego.jpg')
texure = cv2.imread('textures/rice.png')
out = img = cv2.imread('src.jpg')
patch_sz = 10
init_threshold_const = 1


def print_progress(done_patches, total_patches, threshold=0):
    sys.stdout.write('\rPatcheCompleted: %d/%d | Threshold: %.1f'
                     % (done_patches, total_patches, threshold))
    sys.stdout.flush()


def extract_patches(img, patch_sz):
    img = img.astype(np.int32)
    h, w, _ = img.shape
    rh, rw = (h // patch_sz) * patch_sz, (w // patch_sz) * patch_sz
    patches = [
        img[y:y + patch_sz, x:x + patch_sz]
        for y in range(0, h, patch_sz)
        for x in range(0, w, patch_sz)
        if x < rw and y < rh
    ]
    return patches


def extract_texture_patches(tex, patch_sz):
    h, w, _ = tex.shape
    tex = texure.astype(np.int32)
    patches = image.extract_patches_2d(tex, (patch_sz, patch_sz))
    return patches[:h - patch_sz, :w - patch_sz]


def get_best_texture_patch(src_patch, tex_patches, threshold):
    diff = tex_patches - src_patch
    cost = np.sum(diff ** 2, axis=(1, 2, 3)) ** 0.5
    hard_example = np.where(cost < threshold / 2)[0]
    if hard_example.size:
        return hard_example
    return np.where(cost < threshold)


def ind2sub(n, width):
    return n // width, n % width


def fill_output(img_px, output, tex_px, texure):
    x, y = img_px
    tx, ty = tex_px
    tex = texure[tx:tx + patch_sz, ty:ty + patch_sz]
    output[x:x + patch_sz, y:y + patch_sz] = tex


def main():
    patches = extract_patches(img, patch_sz)
    tex_patches = extract_texture_patches(texure, patch_sz)
    for num, patch in enumerate(patches):
        threshold = init_threshold_const * (patch_sz ** 2)

        p_ids = np.array([])
        while not p_ids.size:
            p_ids = get_best_texture_patch(patch, tex_patches, threshold)[0]
            threshold *= 1.1

        p_id = p_ids[randint(0, len(p_ids) - 1)]
        j, i = ind2sub(p_id, texure.shape[1] - patch_sz)
        y, x = ind2sub(num, img.shape[1] // patch_sz)
        fill_output((y * patch_sz, x * patch_sz), out, (j, i), texure)
        # print(num, threshold, len(p_ids))
        print_progress(num, len(patches), threshold)

    cv2.imwrite('output.png', out)

if __name__ == '__main__':
    main()
