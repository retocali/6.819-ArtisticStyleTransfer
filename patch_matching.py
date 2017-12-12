import cv2
import numpy as np
from sys import argv
from sklearn.feature_extraction import image
import time

overlap = 2
thickness = 3
<<<<<<< Updated upstream
threshold = 200
adaptive = True
=======
threshold = 1000000
>>>>>>> Stashed changes

def patch_matching(source, target, patches=(12, 16), quilting=True):
    """ Takes in two filenames as image files
    where the colors from target are moved onto
    source and returns a numpy array as the result"""

    # Split the images into patches
    l, w, c = source.shape
    l /= patches[0]
    w /= patches[1]

    # Find patches of random locations
    if quilting:
        patch_s = image.extract_patches_2d(source, (int(l)+overlap, int(w)+overlap), 2000)
    else:
        patch_s = flatten([np.vsplit(row, patches[0])
                       for row in np.hsplit(source, patches[1])])
    patch_s = rotate_items(patch_s)
    patch_t = flatten([np.vsplit(row, patches[0])
                       for row in np.hsplit(target, patches[1])])

    matches = []

    patch_size = patch_t[0].shape
    print("Patch size:", patch_size)
    for p_t in patch_t:
        # Find the closest patch
        if quilting:
            p_s = min(patch_s, key=lambda x: np.sum((cv2.resize(x, patch_size[:2]) - p_t)**2))
        else:
            p_s = min(patch_s, key=lambda x: np.sum((x - p_t)**2))
        # Add the closest patch
        matches.append(p_s)
    # Concatenate the patches into a image
    print("Concatenating Image")
    cols = []
    if quilting:
        for x in range(patches[1]):
            col = vstack(matches[patches[0] * x:patches[0] * (x + 1)])
            cols.append(col)
        matches = hstack(cols)
        matches = np.fliplr(matches)
        matches = np.rot90(matches,1)
    else:
        matches = [cv2.resize(x, patch_size[:2]) for x in matches]
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
    final = cv2.merge([c1, c2, c3]).astype("uint8")
    print("Final:", final.shape)
    return final

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
        patchworks[patch_size] = image.extract_patches_2d(img, (patch_size, patch_size), 2000)
        patchworks[patch_size] = rotate_items(patchworks[patch_size])
    return patchworks

def quad_tree(input_img, style_img, omega=10, max_patch_size=100, min_patch_size = 25):
    # Read images and convert to color space
    s = cv2.imread(style_img).astype("float32")
    t = cv2.imread(input_img).astype("float32")

    s = cv2.resize(s, (512, 512))
    t = cv2.resize(t, (512, 512))

    # Take measurements of image
    print("Dim:", s.shape, t.shape)
    t_height = t.shape[0]
    t_width = t.shape[1]
    t_height_regions = t_height // max_patch_size
    t_width_regions = t_width // max_patch_size

    # Split the images into patches
    patch_s = create_sub_patches(s, max_patch_size, min_patch_size)
    patch_t = flatten([np.vsplit(row, t_height_regions) for row in np.hsplit(t, t_width_regions)])

    t0 = time.time()
    # Find patches that match most closely
    matches = []
    counter = 0
    step_display = 5
    for p_t in patch_t:
        if counter % step_display == 0:
            print('t = {}'.format(time.time()-t0))
        counter+=1
        ti = p_t.shape[0]
        p_s = min(patch_s[ti], key=lambda x: (np.linalg.norm(np.subtract(x, p_t))**2)/(ti**2))
        d = (np.linalg.norm(np.subtract(p_s, p_t))**2)/(ti**2)
        eta = np.std(p_t)# + d
        print('std: ', np.std(p_t))
        print('d: ', d)
        print('ti: ', ti)
        print('='*20)
        # Split under this condition
        if eta > omega and ti > min_patch_size:
            matches.append(split(p_t, patch_s, omega, min_patch_size))
        
        # Otherwise just add it
        else:
            print('eta: ', eta)
            matches.append(p_s)

    
    # Concatenate the patches into a image
    print("Concatenate")
    cols = []
    for x in range(t_width_regions):
        col = np.vstack(matches[t_height_regions * x:t_height_regions * (x + 1)])
        cols.append(col)
    matches = np.hstack(cols)   
    return matches

def split(patch, style_patches, omega, min_patch_size):
    
    # Base Case Check    
    patch_size = patch.shape[0]
    if patch_size < min_patch_size:
        print('smol')
        return np.full((patch_size, patch_size, 3), 0)
    split_patch = flatten([np.vsplit(row, 2) for row in np.hsplit(patch, 2)])

    # Find patches that match most closely
    matches = []
    ti = patch_size//2
    print('ti splitted: ', ti)
    for sub_patch in split_patch:
        matching_patch = min(style_patches[ti], key=lambda x: (np.linalg.norm(np.subtract(x, sub_patch))**2)/(ti**2))
        d = (np.linalg.norm(np.subtract(matching_patch, sub_patch))**2)/(ti**2)
        eta = np.std(sub_patch)# + d

        # Split under this condition
        if eta > omega and ti > min_patch_size:
            matches.append(split(sub_patch, style_patches, omega, min_patch_size))

        # Otherwise just add it
        else:
            matches.append(matching_patch)
    
    # Concatenate the patches into a image
    cols = []
    for x in range(2):
        col = np.vstack(matches[2 * x:2 * (x + 1)])
        cols.append(col)
    combined_patch = np.hstack(cols)   

    return combined_patch

def rotate_items(array):
    rotated_items = []
    for item in array:
        rotated_items.extend([np.rot90(item, k) for k in range(4)])
    return rotated_items


def hstack(images):
    height = sum(image.shape[0] for image in images)
    real_height = sum(image.shape[0]-overlap for image in images)
    width = max(image.shape[1] for image in images)
    output = np.zeros((height, width, 3))

    y = 0
    old_image = None
    for image in images:
        h, w, d = image.shape
        output[y: y + h, :w] = image
        if (y != 0):
            r = overlap
            t = thickness
            print(np.sum((image[:r, :w] - old_image[h-r:h, :w])**2))
            # if not adaptive or (np.sum((image[:r, :w] - old_image[h-r:h, :w])**2) < threshold):
            for i in range(0, w, 2*thickness):
                if not adaptive or (np.sum((image[:r, :i:i+t] - old_image[h-r:h, :i:i+t])**2) < threshold):
                    output[y-r:y, i:i+t] = image[:r, i:i+t] 
        old_image = image
        y += h
    return cv2.resize(output, (real_height, width))


def vstack(images):
    height = max(image.shape[0] for image in images)
    real_width = sum(image.shape[1]-overlap for image in images)
    width = sum(image.shape[1] for image in images)
    output = np.zeros((height, width, 3))

    x = 0
    old_image = None
    for image in images:
        h, w, d = image.shape
        output[:h, x: x + w] = image
        if (x != 0):
            r = overlap
            t = thickness
            print(np.sum((image[:h, :r] - old_image[:h, w-r:w])**2))
            # if not adaptive (np.sum((image[:h, :r] - old_image[:h, w-r:w])**2) < threshold):
            for i in range(0, w, 2*thickness):        
                if not adaptive or (np.sum((image[i:i+t, :r] - old_image[i:i+t, w-r:w])**2) < threshold):
                    output[i:i+t,x-r:x] = image[i:i+t, :r] 
        old_image = image
        x += h
    return cv2.resize(output, (height, real_width))


def flatten(array):
    return [item for sublist in array for item in sublist]


if __name__ == "__main__":
    if len(argv) < 3:
        print("ERROR: Not enough arguments")
    elif len(argv) == 3:
        # try:
        s = str(argv[1])
        t = str(argv[2])

        im = quad_tree(s, t, omega=15, max_patch_size = 32, min_patch_size = 8)
        print(im.shape)
        im = cv2.resize(im, (500, 400))
        whole = np.hstack((im / 255))
        cv2.namedWindow("results", cv2.WINDOW_NORMAL)
        cv2.imshow('results', whole)
        while cv2.waitKey(0) != q:
            pass
        cv2.destroyAllWindows()

        # except:
        #   print("ERROR: not valid arguments")
    else:
        s = cv2.imread(str(argv[1])).astype("float32")
        t = cv2.imread(str(argv[2])).astype("float32")
        if len(argv) == 4:
            im = patch_matching(s, t, (int(argv[3]), int(argv[3])))
        elif len(argv) == 5:
            im = patch_matching(s, t, (int(argv[3]), int(argv[4])))
        elif len(argv) == 6 and argv[5] == "-qno":
            im = patch_matching(s, t, (int(argv[3]), int(argv[4])), quilting=False)
        else:
            print("ERROR: Too many arguments")

        whole = np.hstack([im / 255])
        cv2.namedWindow("results", cv2.WINDOW_NORMAL)
        cv2.imshow('results', whole)    
        while cv2.waitKey(0) != q:
            pass
        cv2.destroyAllWindows()

