import numpy as np
import h5py
from time import time
import tqdm
# from pyeemd import ceemdan, eemd, emd_num_imfs
# from eofs.standard import Eof
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.pyplot import imread
# from eofs.standard import Eof
# from pyeemd import ceemdan, eemd, emd_num_imfs
from timeit import default_timer as timer
from skimage.util import img_as_float32
from matplotlib.widgets import Slider
import tqdm
# from pyeemd import ceemdan, eemd, emd_num_imfs
# from eofs.standard import Eof
import collections.abc

# 👇️ add attributes to `collections` module
# before you import the package that causes the issue
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.MutableSet = collections.abc.MutableSet
collections.Callable = collections.abc.Callable
from eofs.standard import Eof
from pyeemd import ceemdan, eemd, emd_num_imfs
from timeit import default_timer as timer
#import collections.abc

#collections_abc = getattr(collections, 'abc', collections)
from skimage.measure import block_reduce
from scipy import signal
from scipy.signal import *
from scipy.fftpack import fft, fftfreq, rfft, irfft, ifft

folder='/Volumes/Macintosh HD/seafile_mic/new_work4/'

def get_array_masks(folderul, img_mask_background, img_mask_forground, img_mask_forg_lille):
    imag_bkg = imread(folderul+img_mask_background)
    imag_bkg_flt = imag_bkg.flatten()
    mask_bkg = (imag_bkg_flt > 0)

    imag_for = imread(folderul+img_mask_forground)
    imag_for_flt = imag_for.flatten()
    mask_for = (imag_for_flt > 0)

    imag_for_lille = imread(folderul+img_mask_forg_lille)
    imag_for_flt_lille = imag_for_lille.flatten()
    mask_for_lille = (imag_for_flt_lille > 0)

    return mask_for, mask_bkg, mask_for_lille


def get_mask_red(img_mask_forground):
    imag_for = imread(img_mask_forground)
    down_image = block_reduce(imag_for, block_size=(2, 2), func=np.mean)
    imag_for_flt = down_image.flatten()
    mask_for = (imag_for_flt > 0)
    return mask_for


def get_data(filename, typen, ref):
    with h5py.File(filename, 'r') as fi:
        images = fi['/Images/stabilized_{}_{}'.format(typen, ref)][:]

    # images=images[0:36000]
    # x=images.shape[1]
    # y=images.shape[2]
    # print(images.shape)
    # images=images.reshape((36000,x*y))
    # print(images.shape)
    return images


def mask_array(arr_t, mask):
    arry_mask = arr_t[:, mask]

    return arry_mask


def un_mask_array(dim_x, dim_y, mask, masked_array):
    new_array = np.zeros((36000, dim_x * dim_y))
    new_array[:, mask] = masked_array

    return new_array


def deci_signal(data, factor):
    sig1darr = decimate(data, factor, ftype='iir', axis=0, zero_phase=True)

    return sig1darr


def red_image(imi):
    down_image = block_reduce(imi, block_size=(2, 2), func=np.mean)
    return down_image


# imiges,a,b=get_data('data_img_stab_nestab.h5','/Images/stabilized')
typen = 'affine'
ref = 'mean'
imiges = get_data(folder+'001_data_reduced_stab_ustab.h5', typen, ref)
imiges = imiges.reshape((imiges.shape[0], imiges.shape[1] * imiges.shape[2]))
mask_for, mask_bkg, mask_for_red = get_array_masks(folder,
                                                   'fing_lile_mask_bkg_mean_affine_mean.png',
                                                   'fing_lile_mask_all_finger_mean_affine_mean.png',
                                                   'fing_lile_mask_virf_finger_mean_affine_mean.png')

# mask_for_red=get_mask_red('finger_mask_mean_lille.png')

img_forg = mask_array(imiges, mask_for)
img_bkg = mask_array(imiges, mask_bkg)

print(img_forg.shape)
print(img_bkg.shape)
print(np.mean(img_bkg, axis=1).shape)
img_bkg_mean = np.mean(img_bkg, axis=1)
img_bkg_mean = img_bkg_mean[:, np.newaxis]
img_final = img_forg - img_bkg_mean
# img_final=deci_signal(img_final,2)
# img_final=img_final.reshape((36000,a,b))
print(img_final.shape)
# all_arry=un_mask_array(a,b,mask_for,img_final)
# all_arry=deci_signal(all_arry,2)
# all_arry=all_arry.reshape((18000,a,b))

# all_ar_red=red_image(all_arry[1,:,:])

# print(all_ar_red[:,:].shape)
# print(a)
# print(b)

start = timer()

solver = Eof(img_final)

end = timer()
print('time for EOT solver')
print(end - start)  # Time in seconds, e.g. 5.38091952400282

pcs1 = solver.pcs(npcs=1)
print(pcs1.shape)

eof1 = solver.eofs(neofs=1)
print(eof1.shape)

# reconstruction = solver.reconstructedField([1, 2, 3])
# print(reconstruction.shape)

variance_fractions = solver.varianceFraction()
print(dir(solver))
print(solver._P.shape)
print(solver._flatE.shape)
print(variance_fractions)

cumsum = np.cumsum(variance_fractions)
index = np.where(cumsum < 0.99)
PCnr = np.amax(index) + 1
print(PCnr)
print(index[0][-1])
print(index)
fig1, axs = plt.subplots(1, 1)
# ax=axs.ravel()
print(cumsum.size)
axs.plot(cumsum)
axs.scatter(index, cumsum[index], marker='o', color='r')
axs.set_title('{} PC explain 99\% variance in data'.format(PCnr))
plt.show()

reconstruction1 = solver.reconstructedField(1)
reconstruction2 = solver.reconstructedField(2)
reconstruction3 = solver.reconstructedField(3)

arrayrec1 = np.zeros((18000, imiges.shape[1]), dtype=np.float32)
arrayrec1[:, mask_for] = reconstruction2


# plot_image_timeserie(arrayrec1.reshape((arrayrec1.shape[0],480,640)))

def eemd_wrapper1(imgi):
    imfs = ceemdan(imgi)
    imfst = np.transpose(imfs)
    # print(idx)
    # print(imfst.shape)
    return imfst


col_imf = emd_num_imfs(18000)
# PCnr=5000
print('col_imf')
print(col_imf)
imnR = np.zeros((18000, col_imf, PCnr), dtype='float32')

start = timer()

for i in tqdm.tqdm(range(PCnr)):
    imnR[..., i] = eemd_wrapper1(solver._P[:, i])

print('ImnR.shape is:')
print(imnR.shape)
end = timer()
print('time for CEEMDAN calculations of the {} PCs'.format(PCnr))
print(end - start)  # Time in seconds, e.g. 5.38091952400282
img_eemd = np.zeros((18000, col_imf, img_final.shape[1]), dtype='float32')
print(img_eemd.shape)

start = timer()

modes = [m for m in range(PCnr)]
for ind in tqdm.tqdm(range(col_imf)):
    rval = np.dot(imnR[:, ind, modes], solver._flatE[modes])
    rval = rval.reshape((solver._records,) + solver._originalshape)
    print(rval.shape)
    print('should be equal')
    print(img_eemd[:, ind, :].shape)
    img_eemd[:, ind, :] = rval

end = timer()
print('time for IMF reconstruction from PCs IMFs')
print(end - start)  # Time in seconds, e.g. 5.38091952400282

# img_eemd_final = np.zeros((7200,col_imf,480*640),dtype=np.float32)
# img_eemd_final[:,:,mask_nr] = img_eemd