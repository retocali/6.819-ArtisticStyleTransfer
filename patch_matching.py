import cv2
import numpy as np
from sys import argv
from sklearn.feature_extraction import image


def patch_matching(source, target, patches=(12, 16), sigma=1. / 4):
    """ Takes in two filenames as image files
    where the colors from target are moved onto
    source and returns a numpy array as the result"""

    # add noise to target
    target = target + np.random.normal(0, sigma, np.shape(target))

    # Split the images into patches
    l, w, c = source.shape
    l /= patches[0]
    w /= patches[1]

    # Find patches of random locations
    patch_s = image.extract_patches_2d(
        source, (int(l), int(w)), 2000)
    patch_s = rotate_items(patch_s)

    patch_t = flatten([np.vsplit(row, patches[0])
                       for row in np.hsplit(target, patches[1])])

    matches = []
    for p_t in patch_t:
        # Find the closest patch
        p_s = min(patch_s, key=lambda x: np.sum(np.subtract(x, p_t)**2))
        # Add the closest patch
        matches.append(p_s)
    patch_size = p_s.shape
    # Concatenate the patches into a image
    print("Concatenating Image")
    cols = []
    for x in range(patches[1]):
        col = np.vstack(matches[patches[0] * x:patches[0] * (x + 1)])
        cols.append(col)
    matches = np.hstack(cols)

    # Clip the color values in the channels
    c1, c2, c3 = cv2.split(matches)
    c1 = np.clip(c1, 0, 255)
    c2 = np.clip(c2, 0, 255)
    c3 = np.clip(c3, 0, 255)

    # Add gaussian blur
    return cv2.merge([c1, c2, c3]).astype("uint8")


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

    s = cv2.imread(str(argv[1])).astype("float32")
    t = cv2.imread(str(argv[2])).astype("float32")
    if len(argv) == 3:
        im = patch_matching(s, t)
    elif len(argv) == 4:
        im = patch_matching(s, t, (int(argv[3]), int(argv[3])))
    elif len(argv) == 5:
        im = patch_matching(s, t, (int(argv[3]), int(argv[4])))
    else:
        print("ERROR: Too many arguments")

    # whole = np.hstack([s / 255, t / 255, im / 255])
    cv2.namedWindow("results", cv2.WINDOW_NORMAL)
    cv2.imshow('results', im / 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
