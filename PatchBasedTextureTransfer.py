#!/usr/bin/python
import cv2
import sys
from random import randint

from toolbox import insert_global_vars
from toolbox import get_best_tex_patches, fill_image


img_sample = cv2.imread('textures/rice.png')
out = img = cv2.imread('src.jpg')
[img_height, img_width, _] = img.shape
[sample_height, sample_width, _] = img_sample.shape
PatchSize = patch_sz = 10
init_threshold_const = 1

# Hack!! Send global variables in main script into toolbox
insert_global_vars(locals())


def print_progress(done_patches, total_patches, threshold=0):
    sys.stdout.write('\rPixelsCompleted: %d/%d | Threshold: %.1f'
                     % (done_patches, total_patches, threshold))
    sys.stdout.flush()

done_patches = 0
total_patches = (img_height // patch_sz) * (img_width // patch_sz)
print('start')
print_progress(done_patches, total_patches)

for y in range(0, img_height - patch_sz, patch_sz):
    for x in range(0, img_width - patch_sz, patch_sz):
        done_patches += 1
        threshold = init_threshold_const * patch_sz ** 2

        while True:
            patches_pos = get_best_tex_patches((y, x), threshold)
            if len(patches_pos) > 0:
                break
            threshold *= 1.1

        sample_match = patches_pos[randint(0, len(patches_pos) - 1)]
        fill_image((y, x), sample_match, output=out)

        print_progress(done_patches, total_patches, threshold)

cv2.imwrite('output.png', out)
