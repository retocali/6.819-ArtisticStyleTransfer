from sys import argv
import os
import cv2
import numpy as np
from scipy.cluster.vq import kmeans2, vq

def kmeans_on_image(filename, k=2, itr=10):
    # Get the image
    img = cv2.imread(filename)
    z = img.reshape((-1,3)).astype(float)

    # Run kmeans
    center,dist = kmeans2(z,k,itr)
    code,distance = vq(z,center)

    res = center[code]
    res2 = res.reshape((img.shape))
    cv2.imshow('res2',res2/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    if len(argv) == 1:
        kmeans_on_image('test.jpg');
    elif len(argv) == 2:
        try:
            kmeans_on_image(str(argv[1]))
        except:
            print("ERROR: Not a valid filename")
    else:
        try:
            kmeans_on_image(str(argv[1]), int(argv[2]))
        except:
            print("ERROR: Not valid arguments")
