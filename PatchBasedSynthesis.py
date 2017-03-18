#!/usr/bin/python

import cv2
import sys
import numpy as np
from random import randint

# Image Loading and initializations
InputName = str(sys.argv[1])
img_sample = cv2.imread(InputName)
img_height = 250
img_width = 200
sample_width = img_sample.shape[1]
sample_height = img_sample.shape[0]
img = np.zeros((img_height, img_width, 3), np.uint8)
PatchSize = int(sys.argv[2])
OverlapWidth = int(sys.argv[3])
InitialThresConstant = float(sys.argv[4])

# Picking random patch to begin
randomPatchHeight = randint(0, sample_height - PatchSize)
randomPatchWidth = randint(0, sample_width - PatchSize)
for i in range(PatchSize):
    for j in range(PatchSize):
        img[i, j] = img_sample[randomPatchHeight + i, randomPatchWidth + j]
# Initializating next
GrowPatchLocation = (0, PatchSize)


# ------------------------------------ #
# Best Fit Patch and related functions #
# ------------------------------------ #
def overlap_error_vertical(img_px, sample_px):
    iLeft, jLeft = img_px
    iRight, jRight = sample_px
    overlap_err = 0
    diff = np.zeros((3))
    for i in range(PatchSize):
        for j in range(OverlapWidth):
            for c in range(3):
                diff[c] = int(img[i + iLeft, j + jLeft][c]) - int(
                    img_sample[i + iRight, j + jRight][c])
            overlap_err += (diff[0]**2 + diff[1]**2 + diff[2]**2)**0.5
    return overlap_err


def overlap_error_horizntl(left_px, right_px):
    iLeft, jLeft = left_px
    iRight, jRight = right_px
    overlap_err = 0
    diff = np.zeros((3))
    for i in range(OverlapWidth):
        for j in range(PatchSize):
            for c in range(3):
                diff[c] = int(img[i + iLeft, j + jLeft][c]) - int(
                    img_sample[i + iRight, j + jRight][c])
            overlap_err += (diff[0]**2 + diff[1]**2 + diff[2]**2)**0.5
    return overlap_err


def get_best_patches(px):  # Will get called in GrowImage
    pixels = []
    # check for top layer
    if px[0] == 0:
        for i in range(sample_height - PatchSize):
            for j in range(OverlapWidth, sample_width - PatchSize):
                error = overlap_error_vertical((px[0], px[1] - OverlapWidth),
                                               (i, j - OverlapWidth))
                if error < ThresholdOverlapError:
                    pixels.append((i, j))
                elif error < ThresholdOverlapError / 2:
                    return [(i, j)]
    # check for leftmost layer
    elif px[1] == 0:
        for i in range(OverlapWidth, sample_height - PatchSize):
            for j in range(sample_width - PatchSize):
                error = overlap_error_horizntl((px[0] - OverlapWidth, px[1]),
                                               (i - OverlapWidth, j))
                if error < ThresholdOverlapError:
                    pixels.append((i, j))
                elif error < ThresholdOverlapError / 2:
                    return [(i, j)]
    # for pixel placed inside
    else:
        for i in range(OverlapWidth, sample_height - PatchSize):
            for j in range(OverlapWidth, sample_width - PatchSize):
                error_vertical = overlap_error_vertical(
                    (px[0], px[1] - OverlapWidth), (i, j - OverlapWidth))
                error_horizntl = overlap_error_horizntl(
                    (px[0] - OverlapWidth, px[1]), (i - OverlapWidth, j))
                if error_vertical < ThresholdOverlapError and error_horizntl < ThresholdOverlapError:
                    pixels.append((i, j))
                elif error_vertical < ThresholdOverlapError / 2 and error_horizntl < ThresholdOverlapError / 2:
                    return [(i, j)]
    return pixels


# ------------------------------ #
# Quilting and related Functions #
# ------------------------------ #
def calc_ssd_error(offset, img_px, sample_px):
    err_r = int(
        img[img_px[0] + offset[0], img_px[1] + offset[1]][0]) - int(
            img_sample[sample_px[0] + offset[0], sample_px[1] + offset[
                1]][0])
    err_g = int(
        img[img_px[0] + offset[0], img_px[1] + offset[1]][1]) - int(
            img_sample[sample_px[0] + offset[0], sample_px[1] + offset[
                1]][1])
    err_b = int(
        img[img_px[0] + offset[0], img_px[1] + offset[1]][2]) - int(
            img_sample[sample_px[0] + offset[0], sample_px[1] + offset[
                1]][2])
    return (err_r**2 + err_g**2 + err_b**2) / 3.0


# ---------------- #
# Calculating Cost #
# ---------------- #
def get_cost_vertical(img_px, sample_px):
    cost = np.zeros((PatchSize, OverlapWidth))
    for j in range(OverlapWidth):
        for i in range(PatchSize):
            if i == PatchSize - 1:
                cost[i, j] = calc_ssd_error((i, j - OverlapWidth), img_px,
                                            sample_px)
            else:
                if j == 0:
                    cost[i, j] = calc_ssd_error(
                        (i, j - OverlapWidth), img_px, sample_px) + min(
                            calc_ssd_error((i + 1, j - OverlapWidth),
                                           img_px, sample_px),
                            calc_ssd_error((i + 1, j + 1 - OverlapWidth),
                                           img_px, sample_px))
                elif j == OverlapWidth - 1:
                    cost[i, j] = calc_ssd_error(
                        (i, j - OverlapWidth), img_px, sample_px) + min(
                            calc_ssd_error((i + 1, j - OverlapWidth),
                                           img_px, sample_px),
                            calc_ssd_error((i + 1, j - 1 - OverlapWidth),
                                           img_px, sample_px))
                else:
                    cost[i, j] = calc_ssd_error(
                        (i, j - OverlapWidth), img_px, sample_px) + min(
                            calc_ssd_error((i + 1, j - OverlapWidth),
                                           img_px, sample_px),
                            calc_ssd_error((i + 1, j + 1 - OverlapWidth),
                                           img_px, sample_px),
                            calc_ssd_error((i + 1, j - 1 - OverlapWidth),
                                           img_px, sample_px))
    return cost


def get_cost_horizntl(img_px, sample_px):
    cost = np.zeros((OverlapWidth, PatchSize))
    for i in range(OverlapWidth):
        for j in range(PatchSize):
            if j == PatchSize - 1:
                cost[i, j] = calc_ssd_error((i - OverlapWidth, j), img_px,
                                            sample_px)
            elif i == 0:
                cost[i, j] = calc_ssd_error(
                    (i - OverlapWidth, j), img_px, sample_px) + min(
                        calc_ssd_error(
                            (i - OverlapWidth, j + 1), img_px, sample_px),
                        calc_ssd_error(
                            (i + 1 - OverlapWidth, j + 1), img_px, sample_px))
            elif i == OverlapWidth - 1:
                cost[i, j] = calc_ssd_error(
                    (i - OverlapWidth, j), img_px, sample_px) + min(
                        calc_ssd_error(
                            (i - OverlapWidth, j + 1), img_px, sample_px),
                        calc_ssd_error(
                            (i - 1 - OverlapWidth, j + 1), img_px, sample_px))
            else:
                cost[i, j] = calc_ssd_error(
                    (i - OverlapWidth, j), img_px, sample_px) + min(
                        calc_ssd_error(
                            (i - OverlapWidth, j + 1), img_px, sample_px),
                        calc_ssd_error(
                            (i + 1 - OverlapWidth, j + 1), img_px, sample_px),
                        calc_ssd_error(
                            (i - 1 - OverlapWidth, j + 1), img_px, sample_px))
    return cost


# ------------------------- #
# Finding Minimum Cost Path #
# ------------------------- #


def find_mincost_path_vertical(cost):
    boundary = np.zeros((PatchSize), np.int)
    parent_matrix = np.zeros((PatchSize, OverlapWidth), np.int)
    for i in range(1, PatchSize):
        for j in range(OverlapWidth):
            if j == 0:
                parent_matrix[i, j] = j if cost[i - 1, j] < cost[i - 1, j +
                                                                 1] else j + 1
            elif j == OverlapWidth - 1:
                parent_matrix[i, j] = j if cost[i - 1, j] < cost[i - 1, j -
                                                                 1] else j - 1
            else:
                curr_min = j if cost[i - 1, j] < cost[i - 1, j - 1] else j - 1
                parent_matrix[i, j] = curr_min if cost[i - 1, curr_min] < cost[
                    i - 1, j + 1] else j + 1
            cost[i, j] += cost[i - 1, parent_matrix[i, j]]
    min_idx = 0
    for j in range(1, OverlapWidth):
        min_idx = min_idx if cost[PatchSize - 1, min_idx] < cost[PatchSize - 1,
                                                                 j] else j
    boundary[PatchSize - 1] = min_idx
    for i in range(PatchSize - 1, 0, -1):
        boundary[i - 1] = parent_matrix[i, boundary[i]]
    return boundary


def find_mincost_path_horizntl(cost):
    boundary = np.zeros((PatchSize), np.int)
    parent_matrix = np.zeros((OverlapWidth, PatchSize), np.int)
    for j in range(1, PatchSize):
        for i in range(OverlapWidth):
            if i == 0:
                parent_matrix[i, j] = i if cost[i, j - 1] < cost[i + 1, j -
                                                                 1] else i + 1
            elif i == OverlapWidth - 1:
                parent_matrix[i, j] = i if cost[i, j - 1] < cost[i - 1, j -
                                                                 1] else i - 1
            else:
                curr_min = i if cost[i, j - 1] < cost[i - 1, j - 1] else i - 1
                parent_matrix[i, j] = curr_min if cost[curr_min, j - 1] < cost[
                    i - 1, j - 1] else i + 1
            cost[i, j] += cost[parent_matrix[i, j], j - 1]
    min_idx = 0
    for i in range(1, OverlapWidth):
        min_idx = min_idx if cost[min_idx, PatchSize - 1] < cost[i, PatchSize -
                                                                 1] else i
    boundary[PatchSize - 1] = min_idx
    for j in range(PatchSize - 1, 0, -1):
        boundary[j - 1] = parent_matrix[boundary[j], j]
    return boundary


# -------- #
# Quilting #
# -------- #


def quilt_vertical(boundary, img_px, sample_px):
    for i in range(PatchSize):
        for j in range(boundary[i], 0, -1):
            img[img_px[0] + i, img_px[1] - j] = img_sample[sample_px[
                0] + i, sample_px[1] - j]


def quilt_horizntl(boundary, img_px, sample_px):
    for j in range(PatchSize):
        for i in range(boundary[j], 0, -1):
            img[img_px[0] - i, img_px[1] + j] = img_sample[sample_px[
                0] - i, sample_px[1] + j]


def quilt_patches(img_px, sample_px):
    # check for top layer
    if img_px[0] == 0:
        cost = get_cost_vertical(img_px, sample_px)
        # Getting boundary to stitch
        boundary = find_mincost_path_vertical(cost)
        # Quilting Patches
        quilt_vertical(boundary, img_px, sample_px)
    # check for leftmost layer
    elif img_px[1] == 0:
        cost = get_cost_horizntl(img_px, sample_px)
        # Boundary to stitch
        boundary = find_mincost_path_horizntl(cost)
        # Quilting Patches
        quilt_horizntl(boundary, img_px, sample_px)
    # for pixel placed inside
    else:
        cost_vertical = get_cost_vertical(img_px, sample_px)
        cost_horizntl = get_cost_horizntl(img_px, sample_px)
        boundary_vertical = find_mincost_path_vertical(cost_vertical)
        boundary_horizntl = find_mincost_path_horizntl(cost_horizntl)
        quilt_vertical(boundary_vertical, img_px, sample_px)
        quilt_horizntl(boundary_horizntl, img_px, sample_px)


# ---------------------------- #
# Growing Image Patch-by-patch #
# ---------------------------- #
def fill_image(img_px, sample_px):
    for i in range(PatchSize):
        for j in range(PatchSize):
            img[img_px[0] + i, img_px[1] + j] = img_sample[sample_px[
                0] + i, sample_px[1] + j]


pixelsCompleted = 0
TotalPatches = ((img_height - 1) / PatchSize) * ((img_width) / PatchSize) - 1
sys.stdout.write(
    "Progress : [%-20s] %d%% | PixelsCompleted: %d | ThresholdConstant: --.------"
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
        "Progress : [%-20s] %d%% | PixelsCompleted: %d | ThresholdConstant: %f"
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
