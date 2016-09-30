import ndreg as nd
import SimpleITK as sitk
import numpy as np
import nibabel as nb

inImg = nd.imgDownload("Cocaine174ARACoronal.nii", resolution = 5)
imgArray = sitk.GetArrayFromImage(inImg)

img = nib.Nifti1Image(imgArray, np.eye(4))
nb.save(img, "Cocaine174ARACoronal.nii")


