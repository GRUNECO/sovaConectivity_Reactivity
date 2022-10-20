from cmath import nan
from matplotlib.cbook import flatten
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
import scipy.signal as scsignal
from sovaflow.flow import fit_spatial_filter
from pandas.core.common import flatten

sgl = r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Datos_MsC_Veronica\BIOMARCADORES\sub-CTR001\ses-V1\eeg\sub-CTR001_ses-V1_task-CE_eeg"
file = r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Datos_MsC_Veronica\BIOMARCADORES\derivatives\sovaharmony\sub-CTR001\ses-V1\eeg\sub-CTR001_ses-V1_task-CE_desc-reject[restCE]_eeg"
filename = r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Datos_MsC_Veronica\BIOMARCADORES\derivatives\sovaharmony\sub-CTR001\ses-V1\eeg\sub-CTR001_ses-V1_task-CE_desc-norm_eeg"
raw = mne.io.read_raw_brainvision(sgl + '.vhdr', verbose='error')
preprocessing = mne.read_epochs(file + '.fif', verbose='error')
norm = mne.read_epochs(filename + '.fif', verbose='error')
ch = ['P2','P4','P1','POZ','PO3','PO6','PO7','P7','PO5','O1','P3','P8','OZ','PZ','O2','PO4','PO8','P6','P5']
#correct_montage = copy.deepcopy(ch)
#raw = raw.copy() 
#preprocessing = preprocessing.copy() 
#norm = norm.copy() 
#raw,correct_montage= organize_channels(raw,correct_montage)
#processing,correct_montage= organize_channels(preprocessing,correct_montage)
#norm,correct_montage= organize_channels(norm,correct_montage)

'''fig, ax = plt.subplots(3)
s.plot_psd(ax=ax[0],average=True,fmax=30)
r.plot_psd(ax=ax[1],average=True,fmax=30)
raw.plot_psd(ax=ax[2],average=True,fmax=30)
ax[0].set_title('PSD Original data')
ax[1].set_title('PSD of Preprocessed data')
ax[2].set_title('PSD of Normalized data')
ax[2].set_xlabel('Frequency (Hz)')
fig.set_tight_layout(True)'''

def ch_roi(ch,raw):
    correct_montage = copy.deepcopy(ch)
    raw = raw.copy() 
    raw,correct_montage= organize_channels(raw,correct_montage)
    return raw

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

def welch_pds(signal,roi,continuo=True):
    ch_roi_signal=ch_roi(roi,signal)
    if continuo:
        ics_signal=create_raw_ics(ch_roi_signal)
    else:
        ics_signal=create_raw_ics_c(ch_roi_signal)
    
    ff, Pxx=scsignal.welch(ics_signal.get_data(), fs=signal.info['sfreq'],nperseg=signal.info['sfreq']*2, noverlap=signal.info['sfreq']/2)
    return ff, Pxx

def mean_Pxx_ff(ff,Pxx):
    mean_end = []
    mean_Pxx = []
    for Fr in range(len(ff)):
        for Px in range(len(Pxx)):
            mean_Pxx.append(Pxx[Px][Fr])
        l=list(flatten(mean_Pxx))
        mean_end.append(np.mean(l))
        mean_Pxx = []
    return mean_end

ff_raw, Pxx_raw = welch_pds(raw,ch,continuo=False)
ff_preprocessing, Pxx_preprocessing = welch_pds(preprocessing,ch)
ff_norm, Pxx_norm = welch_pds(norm,ch)
mean_raw = mean_Pxx_ff(ff_raw, Pxx_raw)
mean_preprocessing = mean_Pxx_ff(ff_preprocessing, Pxx_preprocessing)
mean_norm = mean_Pxx_ff(ff_norm, Pxx_norm)

icss=create_raw_ics_c(raw)
icsr=create_raw_ics(preprocessing)
icsraw=create_raw_ics(norm)

fig, ax = plt.subplots(3)
icss.plot_psd(ax=ax[0],average=True,fmin=1,fmax=30,picks=[24])
icsr.plot_psd(ax=ax[1],average=True,fmin=1,fmax=30,picks=[24])
icsraw.plot_psd(ax=ax[2],average=True,fmin=1,fmax=30,picks=[24])
ax[0].set_title('PSD Original data C25')
ax[1].set_title('PSD of Preprocessed data C25')
ax[2].set_title('PSD of Normalized data C25')
ax[2].set_xlabel('Frequency (Hz)')
fig.set_tight_layout(True)

ffs, Pxxs=scsignal.welch(icss.get_data(), fs=s.info['sfreq'],nperseg=s.info['sfreq']*2, noverlap=s.info['sfreq']/2)
ffr, Pxxr=scsignal.welch(icsr.get_data(), fs=r.info['sfreq'],nperseg=r.info['sfreq']*2, noverlap=r.info['sfreq']/2)
ffraw, Pxxraw=scsignal.welch(icsraw.get_data(), fs=raw.info['sfreq'],nperseg=raw.info['sfreq']*2, noverlap=raw.info['sfreq']/2)

def mean_Pxx_ff(ff,Pxx):
    mean_end = []
    mean_Pxx = []
    for Fr in range(len(ff)):
        for Px in range(len(Pxx)):
            mean_Pxx.append(Pxx[Px][Fr])
        l=list(flatten(mean_Pxx))
        mean_end.append(np.mean(l))
        mean_Pxx = []
    return mean_end

mean_s = mean_Pxx_ff(ffs, Pxxs)
mean_r = mean_Pxx_ff(ffr, Pxxr)
mean_raw = mean_Pxx_ff(ffraw, Pxxraw)

f, (ax1, ax2,ax3) = plt.subplots(3, 1,sharex=True, sharey=False)
ax1.set_title('Original')
ax1.plot(ffs,mean_s,color='c')
ax2.set_title('Preprocessing')
ax2.plot(ffr,mean_r,color='m')
ax3.set_title('Normalizate')
ax3.plot(ffraw,mean_raw,color='g')
#plt.legend([ax1, ax2, ax3],["Original", "Preprocessing", "Normalizate"])
#plt.yscale('log')
#plt.ylim((pow(10,-18),pow(10,-4)) )
#plt.yticks(color='w') 
ax1.set_xlim(3,30)
plt.show()





dt = 0.01
plt.psd(x=icss, fs=1000)
plt.psd(x=icsr, fs=1000)
plt.psd(x=icsraw, fs=1000)

fig, ax = plt.subplots(2)
icss.plot(scalings={'eeg':'auto'})
icsr.plot(scalings={'eeg':'auto'})
icsraw.plot(scalings={'eeg':'auto'})
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

