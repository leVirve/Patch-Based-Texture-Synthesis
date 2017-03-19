import numpy as np


def insert_global_vars(vars):
    global img, img_sample
    global overlap, patch_sz
    global sample_height, sample_width
    img, img_sample = vars.get('img'), vars.get('img_sample')
    sample_height, sample_width = vars.get('sample_height'), vars.get('sample_width')
    overlap, patch_sz = vars.get('OverlapWidth'), vars.get('PatchSize')


# ------------------------------------ #
# Best Fit Patch and related functions #
# ------------------------------------ #
def overlap_error_vertical(img_px, sample_px):
    iLeft, jLeft = img_px
    iRight, jRight = sample_px
    img_int = img.astype(np.int32)
    img_sample_int = img_sample.astype(np.int32)

    diff = (img_int[iLeft:iLeft + patch_sz, jLeft:jLeft + overlap]
            - img_sample_int[iRight:iRight + patch_sz, jRight:jRight + overlap])
    overlap_err = np.sum(np.sum(diff**2, axis=2)**0.5)

    return overlap_err


def overlap_error_horizntl(left_px, right_px):
    iLeft, jLeft = left_px
    iRight, jRight = right_px
    img_int = img.astype(np.int32)
    img_sample_int = img_sample.astype(np.int32)

    diff = (img_int[iLeft:iLeft + overlap, jLeft:jLeft + patch_sz]
            - img_sample_int[iRight:iRight + overlap, jRight:jRight + patch_sz])
    overlap_err = np.sum(np.sum(diff**2, axis=2)**0.5)

    return overlap_err


def get_best_tex_patches(px, overlap_err_threshold):
    pixels = []

    def ssd_error(img_pos, tex_pos):
        ix, iy = img_pos
        tx, ty = tex_pos
        src = img.astype(np.int32)
        tex = img_sample.astype(np.int32)
        diff = (src[ix:ix + patch_sz, iy:iy + patch_sz]
                - tex[tx:tx + patch_sz, ty:ty + patch_sz])
        return np.sum(diff ** 2) ** 0.5

    for i in range(patch_sz, sample_height - patch_sz):
        for j in range(patch_sz, sample_width - patch_sz):
            err = ssd_error((px[0], px[1]), (i, j))
            if err < overlap_err_threshold:
                pixels.append((i, j))
            elif err < overlap_err_threshold:
                return [(i, j)]

    return pixels


def get_best_patches(px, overlap_err_threshold):
    pixels = []
    # check for top layer
    if px[0] == 0:
        for i in range(sample_height - patch_sz):
            for j in range(overlap, sample_width - patch_sz):
                error = overlap_error_vertical((px[0], px[1] - overlap),
                                               (i, j - overlap))
                if error < overlap_err_threshold:
                    pixels.append((i, j))
                elif error < overlap_err_threshold / 2:
                    return [(i, j)]
    # check for leftmost layer
    elif px[1] == 0:
        for i in range(overlap, sample_height - patch_sz):
            for j in range(sample_width - patch_sz):
                error = overlap_error_horizntl((px[0] - overlap, px[1]),
                                               (i - overlap, j))
                if error < overlap_err_threshold:
                    pixels.append((i, j))
                elif error < overlap_err_threshold / 2:
                    return [(i, j)]
    # for pixel placed inside
    else:
        for i in range(overlap, sample_height - patch_sz):
            for j in range(overlap, sample_width - patch_sz):
                error_vertical = overlap_error_vertical(
                    (px[0], px[1] - overlap), (i, j - overlap))
                error_horizntl = overlap_error_horizntl(
                    (px[0] - overlap, px[1]), (i - overlap, j))
                if (error_vertical < overlap_err_threshold and
                        error_horizntl < overlap_err_threshold):
                    pixels.append((i, j))
                elif (error_vertical < overlap_err_threshold / 2 and
                      error_horizntl < overlap_err_threshold / 2):
                    return [(i, j)]
    return pixels


# ------------------------------ #
# Quilting and related Functions #
# ------------------------------ #
def calc_ssd_error(offset, img_px, sample_px):
    err_r = int(img[img_px[0] + offset[0], img_px[1] + offset[1]][0]) - int(
        img_sample[sample_px[0] + offset[0], sample_px[1] + offset[1]][0])
    err_g = int(img[img_px[0] + offset[0], img_px[1] + offset[1]][1]) - int(
        img_sample[sample_px[0] + offset[0], sample_px[1] + offset[1]][1])
    err_b = int(img[img_px[0] + offset[0], img_px[1] + offset[1]][2]) - int(
        img_sample[sample_px[0] + offset[0], sample_px[1] + offset[1]][2])
    return (err_r**2 + err_g**2 + err_b**2) / 3.0


# ---------------- #
# Calculating Cost #
# ---------------- #
def get_cost_vertical(img_px, sample_px):
    cost = np.zeros((patch_sz, overlap))
    for j in range(overlap):
        for i in range(patch_sz):
            if i == patch_sz - 1:
                cost[i, j] = calc_ssd_error((i, j - overlap), img_px,
                                            sample_px)
            else:
                if j == 0:
                    cost[i, j] = calc_ssd_error(
                        (i, j - overlap), img_px, sample_px) + min(
                            calc_ssd_error(
                                (i + 1, j - overlap), img_px, sample_px),
                            calc_ssd_error(
                                (i + 1, j + 1 - overlap), img_px, sample_px))
                elif j == overlap - 1:
                    cost[i, j] = calc_ssd_error(
                        (i, j - overlap), img_px, sample_px) + min(
                            calc_ssd_error(
                                (i + 1, j - overlap), img_px, sample_px),
                            calc_ssd_error(
                                (i + 1, j - 1 - overlap), img_px, sample_px))
                else:
                    cost[i, j] = calc_ssd_error(
                        (i, j - overlap), img_px, sample_px) + min(
                            calc_ssd_error(
                                (i + 1, j - overlap), img_px, sample_px),
                            calc_ssd_error(
                                (i + 1, j + 1 - overlap), img_px, sample_px),
                            calc_ssd_error(
                                (i + 1, j - 1 - overlap), img_px, sample_px))
    return cost


def get_cost_horizntl(img_px, sample_px):
    cost = np.zeros((overlap, patch_sz))
    for i in range(overlap):
        for j in range(patch_sz):
            if j == patch_sz - 1:
                cost[i, j] = calc_ssd_error((i - overlap, j), img_px,
                                            sample_px)
            elif i == 0:
                cost[i, j] = calc_ssd_error(
                    (i - overlap, j), img_px, sample_px) + min(
                        calc_ssd_error(
                            (i - overlap, j + 1), img_px, sample_px),
                        calc_ssd_error(
                            (i + 1 - overlap, j + 1), img_px, sample_px))
            elif i == overlap - 1:
                cost[i, j] = calc_ssd_error(
                    (i - overlap, j), img_px, sample_px) + min(
                        calc_ssd_error(
                            (i - overlap, j + 1), img_px, sample_px),
                        calc_ssd_error(
                            (i - 1 - overlap, j + 1), img_px, sample_px))
            else:
                cost[i, j] = calc_ssd_error(
                    (i - overlap, j), img_px, sample_px) + min(
                        calc_ssd_error(
                            (i - overlap, j + 1), img_px, sample_px),
                        calc_ssd_error(
                            (i + 1 - overlap, j + 1), img_px, sample_px),
                        calc_ssd_error(
                            (i - 1 - overlap, j + 1), img_px, sample_px))
    return cost


# ------------------------- #
# Finding Minimum Cost Path #
# ------------------------- #


def find_mincost_path_vertical(cost):
    boundary = np.zeros((patch_sz), np.int)
    parent_matrix = np.zeros((patch_sz, overlap), np.int)
    for i in range(1, patch_sz):
        for j in range(overlap):
            if j == 0:
                parent_matrix[i, j] = j if cost[i - 1, j] < cost[i - 1, j +
                                                                 1] else j + 1
            elif j == overlap - 1:
                parent_matrix[i, j] = j if cost[i - 1, j] < cost[i - 1, j -
                                                                 1] else j - 1
            else:
                curr_min = j if cost[i - 1, j] < cost[i - 1, j - 1] else j - 1
                parent_matrix[i, j] = curr_min if cost[i - 1, curr_min] < cost[
                    i - 1, j + 1] else j + 1
            cost[i, j] += cost[i - 1, parent_matrix[i, j]]
    min_idx = 0
    for j in range(1, overlap):
        min_idx = min_idx if cost[patch_sz - 1, min_idx] < cost[patch_sz - 1,
                                                                j] else j
    boundary[patch_sz - 1] = min_idx
    for i in range(patch_sz - 1, 0, -1):
        boundary[i - 1] = parent_matrix[i, boundary[i]]
    return boundary


def find_mincost_path_horizntl(cost):
    boundary = np.zeros((patch_sz), np.int)
    parent_matrix = np.zeros((overlap, patch_sz), np.int)
    for j in range(1, patch_sz):
        for i in range(overlap):
            if i == 0:
                parent_matrix[i, j] = i if cost[i, j - 1] < cost[i + 1, j -
                                                                 1] else i + 1
            elif i == overlap - 1:
                parent_matrix[i, j] = i if cost[i, j - 1] < cost[i - 1, j -
                                                                 1] else i - 1
            else:
                curr_min = i if cost[i, j - 1] < cost[i - 1, j - 1] else i - 1
                parent_matrix[i, j] = curr_min if cost[curr_min, j - 1] < cost[
                    i - 1, j - 1] else i + 1
            cost[i, j] += cost[parent_matrix[i, j], j - 1]
    min_idx = 0
    for i in range(1, overlap):
        min_idx = min_idx if cost[min_idx, patch_sz - 1] < cost[i, patch_sz -
                                                                1] else i
    boundary[patch_sz - 1] = min_idx
    for j in range(patch_sz - 1, 0, -1):
        boundary[j - 1] = parent_matrix[boundary[j], j]
    return boundary


# -------- #
# Quilting #
# -------- #


def quilt_vertical(boundary, img_px, sample_px):
    for i in range(patch_sz):
        for j in range(boundary[i], 0, -1):
            img[img_px[0] + i, img_px[1] - j] = img_sample[sample_px[0] + i,
                                                           sample_px[1] - j]


def quilt_horizntl(boundary, img_px, sample_px):
    for j in range(patch_sz):
        for i in range(boundary[j], 0, -1):
            img[img_px[0] - i, img_px[1] + j] = img_sample[sample_px[0] - i,
                                                           sample_px[1] + j]


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
def fill_image(img_px, sample_px, output=None):
    x, y = img_px
    ref_x, ref_y = sample_px
    out = output if output is not None else img
    out[x:x + patch_sz, y:y + patch_sz] = \
        img_sample[ref_x:ref_x + patch_sz, ref_y:ref_y + patch_sz]
