#!/usr/bin/python
import cv2
import sys
from random import randint

import numpy as np
from sklearn.feature_extraction import image

texure = cv2.imread('textures/rice.png' if len(sys.argv) <= 2 else sys.argv[1])
source = cv2.imread('src.jpg' if len(sys.argv) <= 2 else sys.argv[2])
output = np.zeros(source.shape, np.uint8)
patch_sz = 10
overlap = 5
init_threshold_const = 3


def print_progress(done_patches, total_patches, threshold=0):
    sys.stdout.write('\rPatcheCompleted: %d/%d | Threshold: %.1f' %
                     (done_patches, total_patches, threshold))
    sys.stdout.flush()


def extract_patches(img, patch_sz, overlap=None):
    h, w, _ = img.shape
    overlap = overlap or patch_sz
    rh, rw = (h // overlap) * overlap, (w // overlap) * overlap
    patches = [(y, x)
               for y in range(0, h - patch_sz, overlap)
               for x in range(0, w - patch_sz, overlap) if x < rw and y < rh]
    return patches


def extract_texture_patches(tex, patch_sz):
    h, w, _ = tex.shape
    tex = tex.astype(np.int32)
    h, w = (h // patch_sz) * patch_sz, (w // patch_sz) * patch_sz
    tex = tex[:h, :w]
    patches = image.extract_patches_2d(tex, (patch_sz, patch_sz))
    return patches, h, w


def get_best_texture_patch(src_patch, tex_patches, out_patch, threshold):
    diff = tex_patches - src_patch
    cost = np.sum(diff**2, axis=(1, 2, 3))**0.5

    h, w, _ = out_patch.shape
    if h and w:
        v_overlap_err = tex_patches[:, :overlap, :] - out_patch[:overlap, :]
        h_overlap_err = tex_patches[:, :, :overlap] - out_patch[:, :overlap]
        overlap_err = np.sum(v_overlap_err ** 2, axis=(1, 2, 3)) ** 0.5 + \
            np.sum(h_overlap_err ** 2, axis=(1, 2, 3)) ** 0.5
        cost = overlap_err * 0.4 + cost * 0.6

    hard_example, = np.where(cost < threshold / 2)
    if hard_example.size:
        return hard_example
    soft_example, = np.where(cost < threshold)
    return soft_example


def ind2sub(n, width):
    return n // width, n % width


def fill_output(img_px, output, tex_px, texure):
    x, y = img_px
    tx, ty = tex_px
    tex = texure[tx:tx + patch_sz, ty:ty + patch_sz]
    output[x:x + patch_sz, y:y + patch_sz] = tex


def main():
    global source
    source = source.astype(np.int32)

    patches = extract_patches(source, patch_sz, 0)
    tex_patches, tex_h, tex_w = extract_texture_patches(texure, patch_sz)

    for num, patch in enumerate(patches):
        b, a = patch
        patch = source[b:b + patch_sz, a:a + patch_sz]
        out_patch = output[b - overlap:b - overlap + patch_sz,
                           a - overlap:a - overlap + patch_sz]

        threshold = init_threshold_const * (patch_sz**2)

        p_ids = np.array([])
        while not p_ids.size:
            p_ids = get_best_texture_patch(patch, tex_patches, out_patch,
                                           threshold)
            threshold *= 1.1

        p_id = p_ids[randint(0, len(p_ids) - 1)]
        j, i = ind2sub(p_id, tex_w - patch_sz + 1)
        y, x = ind2sub(num, source.shape[1] // patch_sz)
        print_progress(num + 1, len(patches), threshold)

        fill_output((y * patch_sz, x * patch_sz), output, (j, i), texure)

    cv2.imwrite('output.png', output)
    cv2.imshow('output', output)
    cv2.waitKey()

if __name__ == '__main__':
    main()
