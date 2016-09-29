import matplotlib as mpl
mpl.use('Agg')

import numpy as np                  # standard Python lib for math ops
import matplotlib.pyplot as plt     # another graphing package

from skimage import data, img_as_float
from skimage import exposure

import cv2
import nibabel as nb

#####################################################

im = nb.load('/home/albert/claritycontrol/code/data/raw/AutA.img')

im = im.get_data()
img = im[:,:,:]

shape = im.shape
#affine = im.get_affine()

x_value = shape[0]
y_value = shape[1]
z_value = shape[2]

#####################################################

imgflat = img.reshape(-1)

img_grey = np.array(imgflat * 255, dtype = np.uint8)

img_eq = exposure.equalize_hist(img_grey)

new_img = img_eq.reshape(x_value, y_value, z_value)
globaleq = nb.Nifti1Image(new_img, np.eye(4))
#nb.save(globaleq, '/home/albert/Thumbo/AutAglobaleq.nii')

######################################################

#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

img_grey = np.array(imgflat * 255, dtype = np.uint8)
#threshed = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)

cl1 = clahe.apply(img_grey)

#cv2.imwrite('clahe_2.jpg',cl1)
#cv2.startWindowThread()
#cv2.namedWindow("adaptive")
#cv2.imshow("adaptive", cl1)
#cv2.imshow("adaptive", threshed)
#plt.imshow(threshed)

localimgflat = cl1 #cl1.reshape(-1)

newer_img = localimgflat.reshape(x_value, y_value, z_value)
localeq = nb.Nifti1Image(newer_img, np.eye(4))
nb.save(localeq, '/home/albert/Thumbo/AutAlocaleq.nii')
