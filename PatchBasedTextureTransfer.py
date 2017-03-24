#!/usr/bin/python
import cv2
import sys
from random import randint

import click
import numpy as np
from sklearn.feature_extraction import image


def print_progress(done_patches, total_patches, threshold=0):
    sys.stdout.write('\rPatch Completed: %d/%d | Threshold: %.1f' %
                     (done_patches, total_patches, threshold))
    sys.stdout.flush()


def extract_patches(img, patch_sz, overlap=None):
    h, w = img.shape[:2]
    overlap = overlap or patch_sz
    rh, rw = (h // overlap) * overlap, (w // overlap) * overlap
    patches = [(y, x)
               for y in range(0, h - patch_sz, overlap)
               for x in range(0, w - patch_sz, overlap) if x < rw and y < rh]
    return patches


def extract_texture_patches(tex, patch_sz):
    h, w = tex.shape[:2]
    tex = tex.astype(np.int32)
    h, w = (h // patch_sz) * patch_sz, (w // patch_sz) * patch_sz
    tex = tex[:h, :w]
    patches = image.extract_patches_2d(tex, (patch_sz, patch_sz))
    return patches, h, w


def get_best_texture_patch(src_patch, tex_patches, out_patch, overlap, threshold):
    diff = tex_patches - src_patch
    axis = tuple(i for i in range(1, len(diff.shape)))
    cost = np.sum(diff**2, axis=axis)**0.5

    h, w = out_patch.shape[:2]
    if overlap and h and w:
        v_overlap_err = tex_patches[:, :overlap, :] - out_patch[:overlap, :]
        h_overlap_err = tex_patches[:, :, :overlap] - out_patch[:, :overlap]
        overlap_err = np.sum(v_overlap_err ** 2, axis=axis) ** 0.5 + \
            np.sum(h_overlap_err ** 2, axis=axis) ** 0.5
        cost = overlap_err * 0.4 + cost * 0.6

    hard_example, = np.where(cost < threshold / 2)
    if hard_example.size:
        return hard_example
    soft_example, = np.where(cost < threshold)
    return soft_example


def ind2sub(n, width):
    return n // width, n % width


def fill_output(img_px, output, tex_px, texure, patch_sz):
    x, y = img_px
    tx, ty = tex_px
    tex = texure[tx:tx + patch_sz, ty:ty + patch_sz]
    output[x:x + patch_sz, y:y + patch_sz] = tex


def get_luminace(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    return ycc[:, :, 0]


def gaussian_blur(img, kernel=(5, 5), sigma=0):
    return cv2.GaussianBlur(img, kernel, sigma)


def texure_transfer(source, texure, output,
                    patch_sz, overlap, threshold_const):
    patches = extract_patches(source, patch_sz, 0)
    tex_patches, tex_h, tex_w = extract_texture_patches(texure, patch_sz)

    for num, patch in enumerate(patches):
        b, a = patch
        patch = source[b:b + patch_sz, a:a + patch_sz]
        out_patch = output[b - overlap:b - overlap + patch_sz,
                           a - overlap:a - overlap + patch_sz]

        threshold = threshold_const * (patch_sz**2)

        p_ids = np.array([])
        while not p_ids.size:
            p_ids = get_best_texture_patch(patch, tex_patches, out_patch,
                                           overlap, threshold)
            threshold *= 1.1

        p_id = p_ids[randint(0, len(p_ids) - 1)]
        j, i = ind2sub(p_id, tex_w - patch_sz + 1)
        y, x = ind2sub(num, source.shape[1] // patch_sz)
        print_progress(num + 1, len(patches), threshold)

        yield j, i, y, x


@click.command()
@click.option('--source', '-s', default='src.jpg')
@click.option('--texure', '-t', default='textures/rice.png')
@click.option('--patch_sz', '-p', default=10)
@click.option('--overlap', '-o', default=5)
@click.option('--threshold_const', '-thr', default=3)
def main(source, texure, patch_sz, overlap, threshold_const):
    img_source = cv2.imread(source)
    img_texure = cv2.imread(texure)
    output = np.zeros(img_source.shape, np.uint8)

    img_source_y = gaussian_blur(get_luminace(img_source)).astype(np.int32)
    img_texure_y = gaussian_blur(get_luminace(img_texure)).astype(np.int32)
    output_y = output[:, :, 0]

    for j, i, y, x in texure_transfer(
            img_source_y, img_texure_y, output_y,
            patch_sz, overlap, threshold_const):
        fill_output(
            (y * patch_sz, x * patch_sz), output,
            (j, i), img_texure,
            patch_sz)
        output_y = output[:, :, 0]

    cv2.imwrite('%s_%s.png' % (source[:-4], texure[:-4]), output)
    cv2.imshow('output', output)
    cv2.waitKey()

if __name__ == '__main__':
    main()
