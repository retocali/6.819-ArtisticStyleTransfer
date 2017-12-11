from sys import argv
import os
import cv2
import numpy as np
from sklearn.feature_extraction import image
from copy import deepcopy


def patch_matching(source, target, patches=(12, 16)):
    """ Takes in two filenames as image files
    where the colors from target are moved onto
    source and returns a numpy array as the result"""

    # Read images and convert to color space
    s = cv2.imread(source).astype("float32")
    t = cv2.imread(target).astype("float32")

    # Split the images into patches
    print("Dim:", s.shape, t.shape)
    # image.extract_patches_2d(s, (600, 800))
    patch_s = flatten([np.vsplit(row, patches[0])
                       for row in np.hsplit(s, patches[1])])
    patch_s = rotate_items(patch_s)
    patch_t = flatten([np.vsplit(row, patches[0])
                       for row in np.hsplit(t, patches[1])])

    matches = []
    print("Patch", patch_t[0].shape, patch_s[0].shape)
    for p_t in patch_t:
        # Find the closest patch
        p_s = min(patch_s, key=lambda x: np.sum(np.subtract(x, p_t)**2))

        # Add the closest patch
        matches.append(p_s)                              

    # Concatenate the patches into a image
    print("Concatenate")
    rows = []
    for x in range(patches[0]):
        rows.append(np.hstack(matches[patches[0] * x:patches[0] * (x + 1)]))
    matches = np.vstack(rows)

    # Clip the color values in the channels
    c1, c2, c3 = cv2.split(matches)
    c1 = np.clip(c1, 0, 255)
    c2 = np.clip(c2, 0, 255)
    c3 = np.clip(c3, 0, 255)

    # Add gaussian noise
    return cv2.merge([c1, c2, c3])

def create_sub_patches(img, max_patch_size, min_patch_size):
    height = img.shape[0]
    width = img.shape[1]

    # Find img split sizes
    patch_sizes = [max_patch_size]
    current_patch_size = max_patch_size
    while current_patch_size > min_patch_size:
        current_patch_size = current_patch_size//2        
        patch_sizes.append(current_patch_size)
    
    patchworks = dict()
    for patch_size in patch_sizes:
        patchworks[patch_size] = list()
        for j in range(0, width, patch_size):
            for i in range(0, height, patch_size):
                patchworks[patch_size].append(img[i:i+patch_size, j:j+patch_size])
    return patchworks

def quad_tree(input_img, style_img, omega=15, max_patch_size=100, min_patch_size = 5):
    # Read images and convert to color space
    s = cv2.imread(style_img).astype("float32")
    t = cv2.imread(input_img).astype("float32")

    # Split the images into patches
    print("Dim:", s.shape, t.shape)
    t_height = t.shape[0]
    t_width = t.shape[1]

    t_height_regions = t_height // max_patch_size
    t_width_regions = t_width // max_patch_size
    print("t_height_regions: ", t_height_regions)
    print("t_width_regions: ", t_width_regions)
    patch_s = create_sub_patches(s, max_patch_size, min_patch_size)
    patch_t = flatten([np.vsplit(row, t_height_regions) for row in np.hsplit(t, t_width_regions)][0])
     
    matches = []
    print('patch_t length: ', len(patch_t))
    for p_t in patch_t:
        ti = p_t.shape[0]
        p_s = min(patch_s[ti], key=lambda x: np.sum(np.subtract(x, p_t)**2))
        d = np.sum(np.subtract(p_s, p_t)**2)
        eta = np.var(p_t) + d
        if False: #eta > omega and ti >= min_patch_size:
            print('hi')
        else:
            matches.append(p_s)

    
      # Concatenate the patches into a image
    print("Concatenate")
    rows = []
    print('shape of matches[0]: ', (matches[0].shape))
    print('length of matches: ', len(matches))
    for x in range(t_width_regions):
        rows.append(np.vstack(matches[t_height_regions * x:t_height_regions * (x + 1)]))
    matches = np.hstack(rows)          

    return matches
    


def rotate_items(array):
    rotated_items = []
    for item in array:
        rotated_items.extend([np.rot90(item, k) for k in range(4)])
    return rotated_items


def flatten(array):
    return [item for sublist in array for item in sublist]


if __name__ == "__main__":
    if len(argv) < 3:
        print("ERROR: Not enough arguments")
    elif len(argv) == 3:
        # try:
        s = str(argv[1])
        t = str(argv[2])
        #im = patch_matching(s, t)
        im = quad_tree(t, s)
        print(im.shape)
        whole = np.hstack((cv2.imread(s) / 255, cv2.imread(t) / 255, im / 255))
        cv2.namedWindow("results", cv2.WINDOW_NORMAL)
        cv2.imshow('results', whole)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # except:
        #   print("ERROR: not valid arguments")
