% Test MATLAB script functionality

clear, clc, close all

dpRoot = fileparts(which('s_CLARITYTractography.m')); % root directory path
cd(dpRoot);
%%
addpath(genpath('NIfTI_20140122')); % add nifti toolbox to matlab path

%% Input parameters
% data info
% store data in daData, each within a folder named 'v??', e.g. v100. each
% data folder contains subfolders named ch0, ch1, brainmask, seedmask etc.
% each subfolder contains tiff images.
dpData = '/Users/Tony/Documents/Git Folder/seelviz/Tony/aut1367/aut1367_raw/'; % data directory path
fnDataArr = dir(fullfile(dpData, 'v*')); % data filenames
chArr = {'ch0', 'ch1'};% ch0 and ch1 are green and red channel, respectively

% parameters
% the scrip loops through all paramerters and saves out each result
% automatically.
dogsigmaArr = [1]; % sigma values for derivitive of gaussian filter, recommended value: 0.6-1.3 (based on actual data)
gausigmaArr = [2.3]; % sigma values for gaussian filer, recommended value: 1.3-2.3 (based on actual data)
angleArr = [25]; % angle threhholds for fiber tracking. recommended value: 20-30

%% Compute structure tensors
disp('*****Start Compute Structure Tensors*****')

for ii = 1    
    % set up result directory
    mkdir('test_MATLAB');
    dpResult = fullfile('test_MATLAB');
    
    for nn = 1 : length(chArr)
        ch = chArr{nn};
        imgstack = single(100 .* randn(100, 100, 100)); % read image stack    

        for jj = 1 : length(dogsigmaArr)
            dogsigma = dogsigmaArr(jj);
            disp(['*****Start DoG Sigma ' num2str(dogsigma) '*****']);

            % generate dog kernels
            dogkercc = single(doggen([dogsigma, dogsigma, dogsigma]));
            dogkerrr = permute(dogkercc, [2, 1, 3]);
            dogkerzz = permute(dogkercc, [1, 3, 2]);

            % compute gradients
            grr = convn(imgstack, dogkerrr, 'same');
            gcc = convn(imgstack, dogkercc, 'same');
            gzz = convn(imgstack, dogkerzz, 'same');

            % compute gradient products
            gprrrr = grr .* grr;
            gprrcc = grr .* gcc;
            gprrzz = grr .* gzz;
            gpcccc = gcc .* gcc;
            gpcczz = gcc .* gzz;
            gpzzzz = gzz .* gzz;

            % compute gradient amplitudes
            ga = sqrt(gprrrr + gpcccc + gpzzzz);
            save_nii(make_nii(ga), fullfile(dpResult, ['test_MATLAB_ga_dogsig' num2str(dogsigma) '.nii']));

            % compute gradient vectors
            gv = cat(4, grr, gcc, gzz);
            gv = gv ./ repmat(ga, [1, 1, 1, 3]);
            save_nii(make_nii(gv), fullfile(dpResult, ['test_MATLAB_gv_dogsig' num2str(dogsigma) '.nii']));

            % compute structure tensors
            for kk = 1 : length(gausigmaArr)
                gausigma = gausigmaArr(kk);
                disp(['*****Start Gauss Sigma ' num2str(gausigma) '*****']);

                dpResultGau = fullfile(dpResult, ['dogsig' num2str(dogsigma) '_gausig' num2str(gausigma)]);
                mkdir(dpResultGau);
                
                % generate gaussian kernel
                gaussker = single(gaussgen([gausigma, gausigma, gausigma]));
                
                % blur gradient products
                gprrrrgauss = convn(gprrrr, gaussker, 'same');
                gprrccgauss = convn(gprrcc, gaussker, 'same');
                gprrzzgauss = convn(gprrzz, gaussker, 'same');
                gpccccgauss = convn(gpcccc, gaussker, 'same');
                gpcczzgauss = convn(gpcczz, gaussker, 'same');
                gpzzzzgauss = convn(gpzzzz, gaussker, 'same');
                
                %%
                % tensors in FSL format
                tensorfsl = cat(4, gprrrrgauss, gprrccgauss, gprrzzgauss, gpccccgauss, gpcczzgauss, gpzzzzgauss);
                fnTensorfsl = ['NEWEST_test_MATLAB_tensorfsl_dogsig' num2str(dogsigma) '_gausig' num2str(gausigma) '.nii'];
                save_nii(make_nii(tensorfsl), fullfile(dpResultGau, fnTensorfsl));
                
                % tensors in DTK format
                tensordtk = cat(4, tensorfsl(:, :, :, 1:2), tensorfsl(:, :, :, 4), tensorfsl(:, :, :, 3), tensorfsl(:, :, :, 5:6));
                fnTensordtk = ['NEWEST_test_MATLAB_tensordtk_dogsig' num2str(dogsigma) '_gausig' num2str(gausigma) '_tensor.nii'];
                save_nii(make_nii(tensordtk), fullfile(dpResultGau, fnTensordtk));
                
            end % gausigma        
        end % dogsigma
    end % channel
end % data