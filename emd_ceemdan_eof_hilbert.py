import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.pyplot import imread
import collections
collections_abc = getattr(collections, 'abc', collections)
from sklearn.preprocessing import MinMaxScaler
from pyeemd import ceemdan, emd_num_imfs
from timeit import default_timer as timer
import tqdm
from scipy import signal
import emd
from sklearn.preprocessing import MinMaxScaler

def find_spikes3(data):
    # meanu_scaled2=minmax_scaling(data, columns=range(0,data.shape[1]))
    # meanu_scaled2=normalize_array(data)
    # meanu_scaled2=minmax_scale(data)
    scaler2 = MinMaxScaler()
    scaler2.fit(data)
    meanu_scaled2 = scaler2.transform(data)
    meanul4 = np.mean(meanu_scaled2, axis=1)
    diffen = np.diff(meanul4, axis=0)
    diffen = np.abs(diffen)
    maxi = np.max(diffen)
    idexing = np.where(diffen > maxi / 3)[0]

    idexing1 = idexing + 1

    repeat_arr = data[idexing + 1, :] - data[idexing, :]

    repeat_arr_cor = np.zeros_like(repeat_arr)

    for i in range(repeat_arr.shape[0]):
        repeat_arr_cor[i, :] = np.sum(repeat_arr[:i + 1, :], axis=0)

    row_toadd = np.zeros_like(repeat_arr_cor[0, :])
    result = np.vstack((row_toadd, repeat_arr_cor))

    ripits = []
    ripits.append(idexing1[0])
    ripits.extend(np.diff(idexing1).tolist())
    ripits.append(data.shape[0] - (idexing1[-1]))
    print('Print ripits')
    print(ripits)
    res1 = np.repeat(result, repeats=ripits, axis=0)
    print(res1.shape)
    print(res1[465:514, :])

    # res2=-res1
    images_r_cor = data - res1
    # meanu_scaled2x=normalize_array(images_r_cor)
    # meanu_scaled2x = minmax_scale(images_r_cor)
    scaler3 = MinMaxScaler()
    scaler3.fit(images_r_cor)
    meanu_scaled2x = scaler3.transform(images_r_cor)
    meanul4x = np.mean(meanu_scaled2x, axis=1)
    diffenx = np.diff(meanul4x, axis=0)
    diffenx = np.abs(diffenx)
    # fig2, axs = plt.subplots(1, 2, sharex=True)
    # ax=axs.ravel()
    # ax[0].plot(data.reshape((data.shape[0],480, 640))[:,229,34])
    # ax[0].scatter(idexing1,data.reshape((data.shape[0],480, 640))[idexing1,229,34], marker='o', color='r')
    # ax[0].plot(images_r_cor.reshape((images_r_cor.shape[0],480, 640))[:,229,34])

    return images_r_cor, idexing1, meanul4, meanul4x, diffen, diffenx


def check_spikes(arr, filename):
    uint_cor, idexing1, meanul4, meanul4x, diffen, _ = find_spikes3(arr.astype(np.float32))
    # uint_cor=uint_cor.reshape((uint_img.shape[0],uint_img.shape[1],uint_img.shape[2]))
    uint_cor = uint_cor.astype(np.float32)

    fig1, axs = plt.subplots(1, 2, sharex=True)
    ax = axs.ravel()

    ax[0].plot(meanul4)
    ax[0].scatter(idexing1, meanul4[idexing1], marker='o', color='r')
    ax[0].plot(meanul4x)
    listtToSr = ' '.join(map(str, idexing1.tolist()))
    ax[0].set_title(listtToSr)
    ax[1].plot(diffen)
    ax[1].scatter(idexing1 - 1, diffen[idexing1 - 1], marker='o', color='r')

    def save_to_file(event, arr, filename):
        with h5py.File(filename + '_nospikes_float32.h5', 'w') as fi:
            grp = fi.create_group("Images")
            grp.create_dataset("Images_nr", data=uint_cor, dtype=np.float32)

    axcut = plt.axes([0.9, 0.0, 0.1, 0.075])
    bcut = Button(axcut, 'Save corr_array to file', color='red', hovercolor='green')
    #   bcut.on_clicked(partial(save_to_file, uint_cor, filename))
    bcut.on_clicked(lambda x: save_to_file(x, uint_cor, filename))
    plt.show()
    return uint_cor


# with h5py.File('001_data_reduced_stab_ustab.h5', 'r') as fi:
#    images = fi['/Images/stabilized_affine_mean'][:]

# 001_data_nonreduced_ustab.h5

with h5py.File('/Users/andreiciubotariu/Downloads/001_data_nonreduced_ustab.h5', 'r') as fi:
    images = fi['/Images/nonstabilized'][:]
# plt.plot(np.mean(images[:,30:40,112:132].flatten()))
imag_bkg = imread('/Users/andreiciubotariu/Downloads/mask_kar.png')
imag_bkg_flt = imag_bkg.flatten()
mask_bkg = (imag_bkg_flt > 0)
imag_bkg1 = imread('/Users/andreiciubotariu/Downloads/mask_kar1.png')
imag_bkg1_flt = imag_bkg1.flatten()
mask_bkg1 = (imag_bkg1_flt > 0)
a = images.shape[1]
b = images.shape[2]
images1 = images.reshape((36000, a * b))
sig = np.mean(images1[:, mask_bkg], axis=1)

images1_cor = check_spikes(images1, 'test1')
scaler = MinMaxScaler()
scaler.fit(images1_cor)
images1_cor=scaler.transform(images1_cor)
sig1 = np.mean(images1_cor[:, mask_bkg], axis=1)
sig2 = np.mean(images1_cor[:, mask_bkg1], axis=1)
# plt.plot(images1_cor.reshape((18000,a,b))[:,30,119])
plt.plot(images1_cor[:, mask_bkg][:, 100])
# print(images[:,30:40,112:132].flatten().shape)
# print(np.mean(images[:,30:40,112:132].reshape((18000,200)),axis=1).shape)
plt.plot(sig1)
plt.plot(sig2)
print(np.array([sig1, sig2]).shape)
# plt.plot((sig1+sig2)/2)
# plt.plot(np.mean(images[:,40:50,210:220].reshape((18000,100)),axis=1))
# plt.plot(np.mean(images[:,3:5,3:5].reshape((18000,4)),axis=1))
# plt.imshow(np.mean(images[:,:,:],axis=0))
# plt.plot(np.mean(images[:,20:40,120:140].reshape((18000,400)),axis=1))
# plt.plot(np.mean(images[:,30:50,112:142].reshape((18000,600)),axis=1)-np.mean(images[:,3:5,3:5].reshape((18000,4)),axis=1))
plt.show()
res = np.array([sig1, sig2])
res = res.transpose()
print(res.shape)


def eemd_wrapper1(imgi):
    imfs = ceemdan(imgi)
    imfst = np.transpose(imfs)
    # print(idx)
    # print(imfst.shape)
    return imfst


col_imf = emd_num_imfs(36000)
# PCnr=5000
print('col_imf')
print(col_imf)
imnR = np.zeros((36000, col_imf, 2), dtype='float32')

start = timer()

for i in tqdm.tqdm(range(2)):
    imnR[..., i] = eemd_wrapper1(res[:, i])

print('ImnR.shape is:')
print(imnR.shape)
end = timer()
# print('time for CEEMDAN calculations of the {} PCs'.format(PCnr))
print(end - start)  # Time in seconds, e.g. 5.38091952400282
# img_eemd = np.zeros((18000, col_imf, img_final.shape[1]),dtype='float32')
# print(img_eemd.shape)

fig1, axs1 = plt.subplots(4, 4, figsize=(15, 6), facecolor='w', edgecolor='k')
fig1.subplots_adjust(hspace=.5, wspace=.001)

axs1 = axs1.ravel()

for i in range(16):
    if i == 0:
        axs1[i].plot(sig1[200:700])
        axs1[i].set_title(str(i))
    else:
        axs1[i].plot(imnR[200:700, i - 1, 0])
        axs1[i].set_title('IMF' + str(i))

fig2, axs2 = plt.subplots(4, 4, figsize=(15, 6), facecolor='w', edgecolor='k')
fig2.subplots_adjust(hspace=.5, wspace=.001)

axs2 = axs2.ravel()

for i in range(16):
    if i == 0:
        axs2[i].plot(sig1)
        axs2[i].set_title(str(i))
    elif i == 6:
        f, Pxx_den = signal.welch(imnR[:, i - 1, 1], 60, nperseg=1024)
        axs2[i].semilogy(f, Pxx_den)
        # axs2[i].ylim([0.5e-3, 1])
        ##axs2[i].xlabel('frequency [Hz]')
        # axs2[i].ylabel('PSD [V**2/Hz]')
        axs2[i].set_title('IMF' + str(i) + str(i + 1))

    else:
        axs2[i].plot(imnR[200:700, i - 1, 1])
        axs2[i].set_title('IMF' + str(i))

plt.show()

sample_rate = 60
seconds = 600
num_samples = sample_rate * seconds

time_vect = np.linspace(0, seconds, num_samples)

IP, IF, IA = emd.spectra.frequency_transform(imnR[:, :, 1], sample_rate, 'hilbert')
IP1, IF1, IA1 = emd.spectra.frequency_transform(imnR[:, :, 0], sample_rate, 'hilbert')
print('IF.shape')
print(IF.shape)
print(IF)
print(IF1)
print(np.allclose(IF, IF1))
print('IA.shape')
print(IA.shape)

freq_range = (0.001, 30, 240, 'log')
f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time=False)
f1, hht1 = emd.spectra.hilberthuang(IF1, IA1, freq_range, sum_time=False)
print(hht.shape)
print(f.shape)

fig = plt.figure(figsize=(10, 6))
# fig.patch.set_facecolor('#E0E0E0')
# Sfig.patch.set_alpha(0.7)
emd.plotting.plot_hilberthuang((hht+hht1)/2, time_vect, f, time_lims=(1, 600), freq_lims=(0.001, 6), fig=fig, log_y=True,
                               cmap='viridis', vmin=0, vmax=0.00004)
#emd.plotting.plot_hilberthuang(hht1, time_vect, f1, time_lims=(1, 600), freq_lims=(0.001, 6), fig=fig, log_y=True,
#                               cmap='viridis', vmin=0, vmax=0.0000002)
plt.show()