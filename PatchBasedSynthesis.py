#!/usr/bin/python

import cv2
import sys
import numpy as np
from random import randint

from toolbox import insert_global_vars
from toolbox import get_best_patches, fill_image, quilt_patches


# Image Loading and initializations
img_sample = cv2.imread(sys.argv[1])
[sample_height, sample_width, _] = img_sample.shape
img_height, img_width = 250, 200
img = np.zeros((img_height, img_width, 3), np.uint8)
PatchSize = int(sys.argv[2])
OverlapWidth = int(sys.argv[3])
InitialThresConstant = float(sys.argv[4])

# Picking random patch to begin
randomPatchHeight = randint(0, sample_height - PatchSize)
randomPatchWidth = randint(0, sample_width - PatchSize)
img[:PatchSize, :PatchSize] = \
    img_sample[randomPatchHeight:randomPatchHeight + PatchSize,
               randomPatchWidth:randomPatchWidth + PatchSize]
# Initializating next
GrowPatchLocation = (0, PatchSize)

pixelsCompleted = 0
TotalPatches = ((img_height - 1) / PatchSize) * ((img_width) / PatchSize) - 1

# Hack!! Send global variables in main script into toolbox
insert_global_vars(locals())


sys.stdout.write(
    "Progress : [%-20s] %d%% | PixelsCompleted: %d | "
    "ThresholdConstant: --.------"
    % ('=' * int(pixelsCompleted * 20 / TotalPatches),
       (100 * pixelsCompleted) / TotalPatches, pixelsCompleted))
sys.stdout.flush()

while GrowPatchLocation[0] + PatchSize < img_height:
    pixelsCompleted += 1
    ThresholdConstant = InitialThresConstant
    # set progress to zer0
    progress = 0
    while progress == 0:
        ThresholdOverlapError = ThresholdConstant * PatchSize * OverlapWidth
        # Get Best matches for current pixel
        List = get_best_patches(GrowPatchLocation)
        if len(List) > 0:
            progress = 1
            # Make A random selection from best fit pxls
            sampleMatch = List[randint(0, len(List) - 1)]
            fill_image(GrowPatchLocation, sampleMatch)
            # Quilt this with in curr location
            quilt_patches(GrowPatchLocation, sampleMatch)
            # upadate cur pixel location
            GrowPatchLocation = (GrowPatchLocation[0],
                                 GrowPatchLocation[1] + PatchSize)
            if GrowPatchLocation[1] + PatchSize > img_width:
                GrowPatchLocation = (GrowPatchLocation[0] + PatchSize, 0)
        # if not progressed, increse threshold
        else:
            ThresholdConstant *= 1.1
    # print pixelsCompleted, ThresholdConstant
    sys.stdout.write('\r')
    sys.stdout.write(
        "Progress : [%-20s] %d%% | PixelsCompleted: %d | "
        "ThresholdConstant: %f"
        % ('=' * int(pixelsCompleted * 20 / TotalPatches),
           (100 * pixelsCompleted) / TotalPatches, pixelsCompleted,
           ThresholdConstant))
    sys.stdout.flush()

# Displaying Images
cv2.imshow('Sample Texture', img_sample)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Generated Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
