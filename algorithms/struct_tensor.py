import numpy as np
import math
from scipy import ndimage
import nibabel as nib
from PIL import Image
from scipy import signal

'''
Function to generate derivatives of Gaussian kernels, in either 1D, 2D, or 3D.
Source code in MATLAB obtained from Qiyuan Tian, Stanford University, September 2015
Edited to work in Python by Tony.
'''

def doggen(sigma):
    halfsize = np.ceil(3 * np.max(sigma))
    x = range(np.single(-halfsize), np.single(halfsize + 1));  # Python colon is not inclusive at end, while MATLAB is.
    dim = len(sigma);
    
    if dim == 1:
        X = np.array(x);  # Remember that, by default, numpy arrays are elementwise multiplicative
        k = -X * np.exp(-X**2/(2 * sigma**2));
        
    elif dim == 2:
        [X, Y] = np.meshgrid(x, x);
        k = -X * np.exp(-X**2/(2*sigma[0]^2) * np.exp(-Y**2))
        
    elif dim == 3:
        [X, Y, Z] = np.meshgrid(x, x, x);
        X = X.transpose(0, 2, 1);  # Obtained through vigorous testing (see below...)
        Y = Y.transpose(2, 0, 1);
        Z = Z.transpose(2, 1, 0);
        
        X = X.astype(float);
        Y = Y.astype(float);
        Z = Z.astype(float);
        k = -X * np.exp(np.divide(-np.power(X, 2), 2 * np.power(sigma[0], 2))) * np.exp(np.divide(-np.power(Y,2), 2 * np.power(sigma[1],2))) * np.exp(np.divide(-np.power(Z,2), 2 * np.power(sigma[2],2)))
        
    else:
        print 'Only supports up to 3 dimensions'
        
    return np.divide(k, np.sum(np.abs(k[:])));



'''
Function to generate Gaussian kernels, in 1D, 2D and 3D.
Source code in MATLAB obtained from Qiyuan Tian, Stanford University, September 2015
Edited to work in Python by Tony. 
'''

def gaussgen(sigma):
    halfsize = np.ceil(3 * max(sigma));
    x = range(np.single(-halfsize), np.single(halfsize + 1));

    dim = len(sigma);

    if dim == 1:
        x = x.astype(float);
        k = np.exp(-x**2 / (2 * sigma^2));
    
    elif dim == 2:
        [X, Y] = np.meshgrid(x, x);
        X = X.astype(float);
        Y = Y.astype(float);
        
        k = np.exp(-X**2 / (2 * sigma[0]**2)) * np.exp(-Y**2 / (2 * sigma[1]**2)); 
    
    elif dim == 3:
        [X, Y, Z] = np.meshgrid(x, x, x);
        X = X.transpose(0, 2, 1);  # Obtained through vigorous testing (see below...)
        Y = Y.transpose(2, 0, 1);
        Z = Z.transpose(2, 1, 0);
        
        X = X.astype(float);  # WHY PYTHON?
        Y = Y.astype(float);
        Z = Z.astype(float);
        k = np.exp(-X**2 / (2 * sigma[0]**2)) * np.exp(-Y**2 / (2 * sigma[1]**2)) * np.exp(-Z**2 / (2 * sigma[2]**2));
    
    else:
        print 'Only supports up to dimension 3'

    return np.divide(k, np.sum(np.abs(k)));


'''
Function takes a single image (TIFF, or other also works), and returns
the single image as a numpy array.  Called by tiff_stack_to_array.
'''
def tiff_to_array(input_path):
    im = Image.open(input_path)
    # im.show()

    imarray = np.array(im)
    # print(imarray)
    
    return imarray


'''
Function takes input_path, which should should lead to a directory.
Loads all TIFFs in input_path, then generates numpy arrays from the
TIFF stack by calling tiff_to_array helper function.
'''
def tiff_stack_to_array(input_path):
    im_list = [];
    for filename in os.listdir(input_path):
        if filename.endswith(".tiff"):
            # print(os.path.join(directory, filename))
            im_arr = tiff_to_array(filename)
            im_list.append(im_arr)
        
    s = np.stack(im_list, axis=2)
    print s.shape
    return s

'''
Function takes a numpy array (from TIFF_stack_to_array) and saves output
FSL structure tensor as filename string. Allows inputting alternate dogsigmaArr,
gausigmaArr, angleArr, although defaults to currently to parameters from MATLAB script.
Also returns tensorfsl (the tensor fsl structure) image numpy array.

## Parameters (the script loops through all parameters and saves each result automatically)
# dogsigmaArr = [1]; Sigma values for derivative of gaussian filter, recommended value: 0.6 - 1.3 (based on actual data)
# gausigmaArr = [2.3]; Sigma values for gaussian filter, recommended value: 1.3 - 2.3 (based on actual data)
# angleArr = [25]; Angle thresholds for fiber tracking, recommended value: 20 - 30.

Follows code from MATLAB CAPTURE scripts, edited to work in Python by Tony.
'''
def generate_FSL_structure_tensor(img_data, filename, dogsigmaArr=[1], gausigmaArr=[2.3], angleArr=[25]):
    for jj in range(len(dogsigmaArr)):
        dogsigma = dogsigmaArr[jj];
        print "Start DoG Sigma on " + str(dogsigma);

        # Generate dog kernels
        dogkercc = doggen([dogsigma, dogsigma, dogsigma]);
        dogkercc = np.transpose(dogkercc, (0, 2, 1));  # annoying

        #print dogkercc.shape;
        #print dogkercc[:, :, 0];

        dogkerrr = np.transpose(dogkercc, (1, 0, 2));

        #print dogkerrr[:, :, 0];
        dogkerzz = np.transpose(dogkercc, (0, 2, 1));

        #print dogkerzz[:, :, 0];

        # Compute gradients
        grr = signal.convolve(img_data, dogkerrr, 'same');

        #print grr[:, :, 0];

        gcc = signal.convolve(img_data, dogkercc, 'same');

        #print gcc[:, :, 0];

        gzz = signal.convolve(img_data, dogkerzz, 'same');

        #print gzz[:, :, 0];

        # Compute gradient products
        gprrrr = np.multiply(grr, grr);

        #print gprrrr[:, :, 0];

        gprrcc = np.multiply(grr, gcc);

        #print gprrcc[:, :, 0];

        gprrzz = np.multiply(grr, gzz);

        #print gprrzz[:, :, 0]

        gpcccc = np.multiply(gcc, gcc);
        gpcczz = np.multiply(gcc, gzz);
        gpzzzz = np.multiply(gzz, gzz);

        # Compute gradient amplitudes
        # print ga.dtype;
        ga = np.sqrt(gprrrr + gpcccc + gpzzzz);

        #print ga[:, :, 0];

        print "GA SHAPE:"
        print ga.shape;

        # Convert numpy ndarray object to Nifti data type
        gradient_amplitudes_data = nib.Nifti1Image(ga, affine=np.eye(4));

        # Save gradient amplitudes image 
        nib.save(gradient_amplitudes_data, 'gradient_amplitudes.nii');

        # Compute gradient vectors
        gv = np.concatenate((grr[..., np.newaxis], gcc[..., np.newaxis], gzz[..., np.newaxis]), axis = 3);
        
        #print gv[:, :, 0, 0];
        
        gv = np.divide(gv, np.tile(ga[..., None], [1, 1, 1, 3]));
        #print gv[:, :, 0, 1];

        print "GV SHAPE:"
        print gv.shape;

        # Convert numpy ndarray object to Nifti data type
        gradient_vectors_data = nib.Nifti1Image(gv, affine=np.eye(4));

        # Save gradient vectors
        nib.save(gradient_vectors_data, 'gradient_vectors.nii');

        # Compute structure tensor
        for kk in range(len(gausigmaArr)):
            gausigma = gausigmaArr[kk];
            print "Start Gauss Sigma with gausigma = " + str(gausigma);

            print "Generating Gaussian kernel..."
            gaussker = np.single(gaussgen([gausigma, gausigma, gausigma]));

            #print gaussker[:, :, 0];

            print "Blurring gradient products..."
            gprrrrgauss = signal.convolve(gprrrr, gaussker, "same");
            
            #print gprrrrgauss[:, :, 0];
            
            gprrccgauss = signal.convolve(gprrcc, gaussker, "same");
            
            #print gprrccgauss[:, :, 0];
            
            gprrzzgauss = signal.convolve(gprrzz, gaussker, "same");
            gpccccgauss = signal.convolve(gpcccc, gaussker, "same");
            gpcczzgauss = signal.convolve(gpcczz, gaussker, "same");
            gpzzzzgauss = signal.convolve(gpzzzz, gaussker, "same");

            print "Saving a copy for this Gaussian sigma..."
            tensorfsl = np.concatenate((gprrrrgauss[..., np.newaxis], gprrccgauss[..., np.newaxis], gprrzzgauss[..., np.newaxis], gpccccgauss[..., np.newaxis], gpcczzgauss[..., np.newaxis], gpzzzzgauss[..., np.newaxis]), axis = 3);
            
             # Convert numpy ndarray object to Nifti data type
            tensor_fsl_data = nib.Nifti1Image(tensorfsl, affine=np.eye(4));

            nib.save(tensor_fsl_data, str(filename) + "dogsigma_" + str(jj) + "gausigma_" + str(kk) + 'tensorfsl.nii');

    print 'Completed computing on ' + str(filename) + '!'
    return tensorfsl

