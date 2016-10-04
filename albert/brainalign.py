import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ndreg import *
import ndio.remote.neurodata as neurodata
import nibabel as nb

refToken = "ara_ccf2"
refImg = imgDownload(refToken)
imgShow(refImg)
plt.savefig("refImg_initial.png", bbox_inches='tight')

imgShow(refImg, vmax=500)
plt.savefig("refImg_initial_vmax500.png", bbox_inches='tight')

refAnnoImg = imgDownload(refToken, channel="annotation")
imgShow(refAnnoImg, vmax=1000)
plt.savefig("refAnnoImg_initial_vmax1000.png", bbox_inches='tight')

randValues = np.random.rand(1000,3)
randValues = np.concatenate(([[0,0,0]],randValues))
randCmap = matplotlib.colors.ListedColormap (randValues)
imgShow(refAnnoImg, vmax=1000, cmap=randCmap)
plt.savefig("ColorefAnnoImg_initial_vmax1000.png", bbox_inches='tight')

imgShow(refImg, vmax=500, newFig=False)
imgShow(refAnnoImg, vmax=1000, cmap=randCmap, alpha=0.2, newFig=False)
plt.show()
plt.savefig("OverlaidImg.png", bbox_inches='tight')

inToken = "Control258"
nd = neurodata()

inImg = imgDownload(inToken, resolution=5)
imgShow(inImg, vmax=500)
plt.savefig("rawImgvmax500.png", bbox_inches='tight')

inImg.SetSpacing([0.01872, 0.01872, 0.005])
inImg_download = inImg

inImg = imgResample(inImg, spacing=refImg.GetSpacing())
imgShow(inImg, vmax=500)
plt.savefig("resample_inImg.png", bbox_inches='tight')

inImg = imgReorient(inImg, "LAI", "RSA")
imgShow(inImg, vmax=500)
plt.savefig("resample_inImg_rotated.png", bbox_inches='tight')

inImg_reorient = inImg

spacing=[0.25,0.25,0.25]
refImg_ds = imgResample(refImg, spacing=spacing)
imgShow(refImg_ds, vmax=500)
plt.savefig("resample_refImg.png", bbox_inches='tight')

inImg_ds = imgResample(inImg, spacing=spacing)
imgShow(inImg_ds, vmax=500)
plt.savefig("inImg_ds.png", bbox_inches='tight')

affine = imgAffineComposite(inImg_ds, refImg_ds, iterations=100, useMI=True, verbose=True)

inImg_affine = imgApplyAffine(inImg, affine, size=refImg.GetSize())
imgShow(inImg_affine, vmax=500)
plt.savefig("inImg_affine.png", bbox_inches='tight')

inImg_ds = imgResample(inImg_affine, spacing=spacing)
(field, invField) = imgMetamorphosisComposite(inImg_ds, refImg_ds, alphaList=[0.05, 0.02, 0.01], useMI=True, iterations=100, verbose=True)
inImg_lddmm = imgApplyField(inImg_affine, field, size=refImg.GetSize())
imgShow(inImg_lddmm, vmax = 500)

imgShow(inImg_lddmm, vmax=500, newFig=False, numSlices=1)
imgShow(refAnnoImg, vmax=1000, cmap=randCmap, alpha=0.2, newFig=False, numSlices=1)
plt.savefig("overlay.png", bbox_inches='tight')

##################
# Reverse orientation
########

invAffine = affineInverse(affine)
invAffineField = affineToField(invAffine, refImg.GetSize(), refImg.GetSpacing())
invField = fieldApplyField(invAffineField, invField)
inAnnoImg = imgApplyField(refAnnoImg, invField,useNearest=True, size=inImg_reorient.GetSize())
imgShow(inAnnoImg, vmax=1000, cmap=randCmap)
plt.savefig("reverse_affine_annotations.png", bbox_inches='tight')

inAnnoImg = imgReorient(inAnnoImg, "RSA", "LAI")
imgShow(inAnnoImg, vmax=1000, cmap=randCmap)
plt.savefig("reoriented_reverse_affine_annotation.png", bbox_inches='tight')

inAnnoImg = imgResample(inAnnoImg, spacing=inImg_download.GetSpacing(), size=inImg_download.GetSize(), useNearest=True)
imgShow(inImg_download, vmax=500, numSlices=1, newFig=False)
imgShow(inAnnoImg, vmax=1000, cmap=randCmap, alpha=0.2, numSlices=1, newFig=False)
plt.savefig("final_atlas.png", bbox_inches='tight')

imgWrite(inAnnoImg, "final_resized_atlas.nii")


