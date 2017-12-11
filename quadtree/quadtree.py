import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
np.set_printoptions(threshold=np.nan)

input_img = scipy.misc.imread('./images/input.png')
style_img = scipy.misc.imread('./images/style.png')

height = style_img.shape[0]
top = height//2-140
style_img = style_img[top:top+280, :]



#def gen_quad_tree(img, max_patch_size = 20, min_patch_size = 5):
    

def gen_patch_inds(img, patch_size=20, patch_display=True):
    height = img.shape[0]
    width = img.shape[1]
    patch_array = np.zeros((height,width), dtype=int)
    current_ind = 0
    for j in range(0, height, patch_size):
        for i in range(0, width, patch_size):
            fill_array = np.full((patch_size, patch_size), current_ind)
            patch_array[i:i+patch_size, j:j+patch_size] = fill_array
            current_ind += 1
    return patch_array

plt.figure(1)
plt.subplot(121)
plt.imshow(style_img)
plt.subplot(122)
plt.imshow(input_img)

patches = gen_patch_inds(style_img)
nrm = colors.Normalize(vmin=0, vmax=255)
mapped_patches = nrm(patches)
plt.figure(2)
plt.imshow(mapped_patches, cmap='jet')
plt.show()
