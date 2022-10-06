from cmath import nan
import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display_html, display
import collections
from sovaflow.flow import preflow, organize_channels
import mne
import copy
from sovaflow.utils import createRaw
from sovaharmony.postprocessing import get_spatial_filter,get_ics_power_derivatives
import scipy.signal as signal
from sovaflow.flow import fit_spatial_filter

BIO_channels=pd.read_feather(r'E:\Academico\Universidad\Posgrado\Tesis\Paquetes\sovaConectivity_Reactivity\longitudinal_data_powers_long_CE_norm_channels.feather')
BIO_components=pd.read_feather(r'E:\Academico\Universidad\Posgrado\Tesis\Paquetes\sovaConectivity_Reactivity\longitudinal_data_powers_long_CE_norm_components.feather')
sgl = r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Datos_MsC_Veronica\BIOMARCADORES\sub-CTR001\ses-V1\eeg\sub-CTR001_ses-V1_task-CE_eeg"
file = r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Datos_MsC_Veronica\BIOMARCADORES\derivatives\sovaharmony\sub-CTR001\ses-V1\eeg\sub-CTR001_ses-V1_task-CE_desc-reject[restCE]_eeg"
filename = r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Datos_MsC_Veronica\BIOMARCADORES\derivatives\sovaharmony\sub-CTR001\ses-V1\eeg\sub-CTR001_ses-V1_task-CE_desc-norm_eeg"
s = mne.io.read_raw_brainvision(sgl + '.vhdr', verbose='error')
r = mne.read_epochs(file + '.fif', verbose='error')
raw = mne.read_epochs(filename + '.fif', verbose='error')
ch_post = ['P2','P4','P1','POZ','PO3','PO6','PO7','P7','PO5','O1','P3','P8','OZ','PZ','O2','PO4','PO8','P6','P5']
correct_montage = copy.deepcopy(ch_post)
raw = raw.copy() 
r = r.copy() 
s = s.copy() 
raw,correct_montage= organize_channels(raw,correct_montage)
r,correct_montage= organize_channels(r,correct_montage)
s,correct_montage= organize_channels(s,correct_montage)

'''fig, ax = plt.subplots(3)
s.plot_psd(ax=ax[0],average=True,fmax=30)
r.plot_psd(ax=ax[1],average=True,fmax=30)
raw.plot_psd(ax=ax[2],average=True,fmax=30)
ax[0].set_title('PSD Original data')
ax[1].set_title('PSD of Preprocessed data')
ax[2].set_title('PSD of Normalized data')
ax[2].set_xlabel('Frequency (Hz)')
fig.set_tight_layout(True)'''

def create_raw_ics(raw_epo):
    def_spatial_filter='58x25'
    spatial_filter = get_spatial_filter(def_spatial_filter)
    signal_epo = raw_epo.copy()
    bands_labels = ['delta', 'theta', 'alpha-1', 'alpha-2', 'beta1', 'beta2','beta3', 'gamma']
    A,W,spatial_filter_chs = spatial_filter
    intersection_chs = list(set(spatial_filter_chs).intersection(signal_epo.ch_names))
    W_adapted = fit_spatial_filter(W,spatial_filter_chs,intersection_chs,mode='demixing')
    signal_epo.reorder_channels(intersection_chs)
    (e, c, t) = signal_epo._data.shape
    signalCont = np.reshape(np.transpose(signal_epo.get_data(),(1,2,0)),(c,t*e),order='F')
    ics = W_adapted @ signalCont
    for e in range(signal_epo._data.shape[0]):
        for c in range(signal_epo._data.shape[1]):
            assert np.all(signal_epo._data[e,c,:] == signalCont[c,e*t:(e+1)*t])
    signal_ics = createRaw(ics,signal_epo.info['sfreq']) 
    return signal_ics

def create_raw_ics_c(raw_epo):
    def_spatial_filter='58x25'
    spatial_filter = get_spatial_filter(def_spatial_filter)
    signal_epo = raw_epo.copy()
    bands_labels = ['delta', 'theta', 'alpha-1', 'alpha-2', 'beta1', 'beta2','beta3', 'gamma']
    A,W,spatial_filter_chs = spatial_filter
    intersection_chs = list(set(spatial_filter_chs).intersection(signal_epo.ch_names))
    W_adapted = fit_spatial_filter(W,spatial_filter_chs,intersection_chs,mode='demixing')
    signal_epo.reorder_channels(intersection_chs)
    ics = W_adapted @ signal_epo.get_data()
    signal_ics = createRaw(ics,signal_epo.info['sfreq']) 
    return signal_ics

icss=create_raw_ics_c(s)
icsr=create_raw_ics(r)
icsraw=create_raw_ics(raw)

fig, ax = plt.subplots(3)
icss.plot_psd(ax=ax[0],average=True,fmin=1,fmax=30,picks=[24])
icsr.plot_psd(ax=ax[1],average=True,fmin=1,fmax=30,picks=[24])
icsraw.plot_psd(ax=ax[2],average=True,fmin=1,fmax=30,picks=[24])
ax[0].set_title('PSD Original data C25')
ax[1].set_title('PSD of Preprocessed data C25')
ax[2].set_title('PSD of Normalized data C25')
ax[2].set_xlabel('Frequency (Hz)')
fig.set_tight_layout(True)

ffs, Pxxs=signal.welch(s.get_data(), fs=s.info['sfreq'],nperseg=s.info['sfreq']*2, noverlap=s.info['sfreq']/2)
ffr, Pxxr=signal.welch(r._data[1], fs=r.info['sfreq'],nperseg=r.info['sfreq']*2, noverlap=r.info['sfreq']/2)
ffraw, Pxxraw=signal.welch(raw._data[1], fs=raw.info['sfreq'],nperseg=raw.info['sfreq']*2, noverlap=raw.info['sfreq']/2)

mean_Pxxs = []
mean_end = []
for j in range(len(ffs)):
    for i in range(len(Pxxs)):
        mean_Pxxs.append(Pxxs[i][j])
        mean_end.append(mean_Pxxs)
        mean_Pxxs = []
    if len(mean_end) > 1:
       mean_end.append(mean_end.mean())

plt.plot(ffs,mean_end)
plt.plot(ffr,Pxxr)
plt.plot(ffraw,Pxxraw)

dt = 0.01
plt.psd(s.get_data(), s.info['sfreq'],dt)
plt.psd(r._data[1], r.info['sfreq'],dt)
plt.psd(raw._data[1], raw.info['sfreq'],dt)

fig, ax = plt.subplots(2)
r.ax.plot(scalings={'eeg':'auto'})
raw.plot(scalings={'eeg':'auto'})
ax[0].set_title('Wica')
ax[1].set_title('preprocessing and normalization')
ax[1].set_xlabel('Frequency (Hz)')
fig.set_tight_layout(True)


delta=BIO_channels[(BIO_channels['Bands']=='delta') & (BIO_channels['Channels']=='P2')]['Powers']
theta=BIO_channels[BIO_channels['Bands']=='theta']['Powers']
alpha1=BIO_channels[BIO_channels['Bands']=='alpha-1']['Powers']
beta1=BIO_channels[BIO_channels['Bands']=='beta1']['Powers']
beta2=BIO_channels[BIO_channels['Bands']=='beta2']['Powers']
beta3=BIO_channels[BIO_channels['Bands']=='beta3']['Powers']
alpha2 = []
for ch in ch_post:
    alpha2.append(BIO_channels[(BIO_channels['Bands']=='alpha-2') & (BIO_channels['Channels']==ch)]['Powers'])

