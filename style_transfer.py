from sys import argv
import cv2
import numpy as np
from patch_matching import patch_matching, quad_tree, create_sub_patches
from color_transfer import color_transfer_fast


def style_transfer(source, target, kernel=(11, 11), weight=0.5):
    # Color transfer the image
    shape = target.shape[:2]
    t_c = color_transfer_fast(source, target)
    print("Color transfered")

    # Produce the hallucination from the blur image]
    hallucination = cv2.resize(patch_matching(source, t, (40, 50)), shape[::-1], 0)
    print("Hallucination completed:", hallucination.shape)
    final = color_transfer_fast(target, hallucination)
    # Compute a Weighted Average
    return weighted_average(t_c, hallucination, weight), weighted_average(target, final, weight)

def style_transfer_quad(source, target, kernel=(11, 11), weight=0.25):
    # Color transfer the image
    shape = target.shape[:2]
    t_c = color_transfer_fast(source, target)
    print("Color transfered")

    # Produce the hallucination from the blur image]
    hallucination = cv2.resize(quad_tree(source.astype("float32"), t.astype("float32"), omega=15, max_patch_size = 32,  min_patch_size = 8) ,shape[::-1], 0)
    print("Hallucination completed:", hallucination.shape)
    final = color_transfer_fast(target, hallucination)
    # Compute a Weighted Average
    return weighted_average(t_c, hallucination, weight), weighted_average(target, final, weight)


def weighted_average(im1, im2, weight):
    return np.add(np.multiply(im1, weight), np.multiply(im2, 1 - weight))


if __name__ == "__main__":
    if len(argv) < 3:
        print("ERROR: Not enough arguments")
    elif len(argv) == 3:
        # try:
        s = cv2.imread(str(argv[1]))
        t = cv2.imread(str(argv[2]))
        im, im2 = style_transfer(s, t)
        print(im.shape)
        whole = np.hstack(
            (s / 255, t / 255, im / 255, im2 / 255))
        cv2.namedWindow("results", cv2.WINDOW_NORMAL)
        cv2.imshow('results', whole)
        while cv2.waitKey(33) != ord('k'):
            pass
        cv2.destroyAllWindows()
        # except:
        #     print("ERROR: not valid arguments")
    elif len(argv) == 4 and argv[1] == "--quad":
        # try:
        s = cv2.imread(str(argv[2]))
        t = cv2.imread(str(argv[3]))
        im, im2 = style_transfer_quad(s, t)
        print(im.shape)
        whole = np.hstack(
            (s / 255, t / 255, im / 255, im2 / 255))
        cv2.namedWindow("results", cv2.WINDOW_NORMAL)
        cv2.imshow('results', whole)
        while cv2.waitKey(33) != ord('k'):
            pass
        cv2.destroyAllWindows()
        # except:
        #     print("ERROR: not valid arguments")
