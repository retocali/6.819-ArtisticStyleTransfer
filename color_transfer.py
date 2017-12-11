from sys import argv
import os
import numpy as np
import cv2
from scipy.cluster.vq import kmeans, vq


# Super fast color transfer algorithm using only means and standard deviation based on
# https://github.com/jrosebr1/color_transfer
def color_transfer_fast(source, target, colorspace="LAB"):
    """ Takes in two filenames as mage files
    where the colors from target are moved onto
    source and returns a numpy array as the result"""

    # Read images and convert to color space
    if colorspace == "RGB":
        pass
    elif colorspace == "LAB":
        s = cv2.cvtColor(source,
                         cv2.COLOR_BGR2LAB).astype("float32")
        t = cv2.cvtColor(target,
                         cv2.COLOR_BGR2LAB).astype("float32")
    elif colorspace == "YCrCb":
        s = cv2.cvtColor(source,
                         cv2.COLOR_BGR2YCrCb).astype("float32")
        t = cv2.cvtColor(target,
                         cv2.COLOR_BGR2YCrCb).astype("float32")
    else:
        raise NameError

    m1_t, s1_t, m2_t, s2_t, m3_t, s3_t = im_mean_and_std(t)
    m1_s, s1_s, m2_s, s2_s, m3_s, s3_s = im_mean_and_std(s)

    # Compute the new channels by subtracting
    # the mean of each channel and scaling it by each
    # image's standard deviation then adding the new mean
    c1, c2, c3 = cv2.split(t)
    print(s1_t / s1_s, s2_t / s2_s, s3_t / s3_s, s1_t, s2_t, s3_t)
    c1 = (c1 - m1_t) * (s1_s / s1_t) + m1_s
    c2 = (c2 - m2_t) * (s2_s / s2_t) + m2_s
    c3 = (c3 - m3_t) * (s3_s / s3_t) + m3_s

    # Cap off colors greater than 255
    c1 = np.clip(c1, 0, 255)
    c2 = np.clip(c2, 0, 255)
    c3 = np.clip(c3, 0, 255)

    # Fuse channels and convert to BGR
    new_image = cv2.merge([c1, c2, c3])
    if colorspace == "RGB":
        return new_image.astype("uint8")
    if colorspace == "LAB":
        return cv2.cvtColor(new_image.astype("uint8"), cv2.COLOR_LAB2BGR)
    if colorspace == "YCrCb":
        return cv2.cvtColor(new_image.astype("uint8"), cv2.COLOR_YCrCb2BGR)


def color_transfer(source, target, k=5, kernel=(11, 11), _min=1e-16):
    """ Takes in two filenames as image files
    where the colors from target are moved onto
    source and returns a numpy array as the result
    Uses kmeans clustering on the above algorithm"""

    # Read images and convert to lab
    s = cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2LAB).astype("float32")
    t = cv2.cvtColor(cv2.imread(target), cv2.COLOR_BGR2LAB).astype("float32")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ___, label_s, center_s = cv2.kmeans(
        cv2.GaussianBlur(s, kernel, 0).flatten(), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    ___, label_t, center_t = cv2.kmeans(
        cv2.GaussianBlur(t, kernel, 0).flatten(), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    new_image = np.zeros_like(s)
    label_s = np.reshape(label_s, s.shape)
    label_t = np.reshape(label_t, t.shape)
    clusters = list(range(k))
    print("S:", center_s, "\nT:", center_t)

    # Slice the image into mask of labels and color transfer each cluster
    for l_s in range(k):
        # Make masks by finding k values with smallest distance in colors
        l_t = min(clusters,
                  key=lambda x: np.sum((center_t[x] - center_s[l_s])**2))
        clusters.remove(l_t)
        mask_s = np.equal(label_s, l_s)
        mask_t = np.equal(label_t, l_t)
        # Produce the slices
        t_slice = np.multiply(t, mask_t)
        s_slice = np.multiply(s, mask_s)
        m1_t, s1_t, m2_t, s2_t, m3_t, s3_t = im_mean_and_std(t_slice)
        m1_s, s1_s, m2_s, s2_s, m3_s, s3_s = im_mean_and_std(s_slice)

        # Compute the new channels by subtracting
        # the mean of each channel and scaling it by each
        # image's standard deviation then adding the new mean
        c1, c2, c3 = cv2.split(t_slice)

        c1 = (c1 - m1_t) * (s1_s / s1_t) + \
            m1_s if s1_s != 0 and s1_t != 0 else c1
        c2 = (c2 - m2_t) * (s2_s / s2_t) + \
            m2_s if s2_s != 0 and s2_t != 0 else c2
        c3 = (c3 - m3_t) * (s3_s / s3_t) + \
            m3_s if s3_s != 0 and s3_t != 0 else c3

        # Cap off colors greater than 255
        c1 = np.clip(c1, 0, 255)
        c2 = np.clip(c2, 0, 255)
        c3 = np.clip(c3, 0, 255)

        # Fuse channels and convert to BGR
        new_slice = cv2.merge([c1, c2, c3])
        new_image = np.add(new_image, new_slice)
    return cv2.cvtColor(new_image.astype("uint8"), cv2.COLOR_LAB2BGR)


def im_mean_and_std(im):
    """Takes numpy array in LAB
    space and returns the mean and
    std for each channel as a tuple"""
    l, a, b = cv2.split(im)
    return (l.mean(), l.std(), a.mean(), a.std(), b.mean(), b.std())


if __name__ == "__main__":
    if len(argv) < 3:
        print("ERROR: Not enough arguments")
    elif len(argv) == 3:
        try:
            print("This implementation performs poorly. Add --fast for a better one")
            s = cv2.imread(str(argv[1]))
            t = cv2.imread(str(argv[2]))
            im = color_transfer(s, t)

            whole = np.hstack([s, t, im])
            cv2.namedWindow("results", cv2.WINDOW_NORMAL)
            cv2.imshow('results', whole)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except:
            print("ERROR: not valid arguments")
    elif len(argv) == 5:
        try:
            if str(argv[1]) != "--fast":
                raise NameError

            s = cv2.imread(str(argv[2]))
            t = cv2.imread(str(argv[3]))
            colorspace = str(argv[4])
            im = color_transfer_fast(s, t, colorspace)

            whole = np.hstack([s, t, im])
            cv2.namedWindow("results", cv2.WINDOW_NORMAL)
            cv2.imshow('results', whole)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except:
            print("ERROR: not valid arguments")
    elif len(argv) == 4:
        try:
            if str(argv[1]) != "--fast":
                raise NameError

            s = cv2.imread(str(argv[2]))
            t = cv2.imread(str(argv[3]))
            im = color_transfer_fast(s, t)

            whole = np.hstack([s, t, im])
            cv2.namedWindow("results", cv2.WINDOW_NORMAL)
            cv2.imshow('results', whole)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("ERROR: not valid arguments")
    else:
        print("ERROR: not valid arguments")
