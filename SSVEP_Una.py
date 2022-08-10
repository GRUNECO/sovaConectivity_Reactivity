import mne
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal;

#Importación de un EDF
fnameE="C:\Sujetos\Mauro\PIS"
raw= mne.io.read_raw_edf(fnameE,preload=True) #Función de lector para datos FIF sin procesar.
#los datos se precargarán en la memoria (rápido, requiere una gran cantidad de memoria).
raw.plot(duration=10.0,start=20.0,n_channels=10,title='EEG Original')
#disenando usando la tabla

# Datos crudos 
data, times = raw[:, :] # Obtiene datos crudos de la señal
data =data*1000000 # Se escala con fines comparativos con matlab 
totalData=len(times) 

plt.plot(times,data[0,:]);
plt.ylabel('Amplitud ');
plt.xlabel('Tiempo');
plt.title('Senal');
plt.grid(True)
#plt.xlim([0,30]);
plt.show()

#%%

low_fre=1 # Frecuencia baja del filtro
high_frec=40 # Frecuencia alta del filtro 

#Filtrado entre 1 y 100 Hz
raw=raw.filter(low_fre,high_frec,fir_design='firwin', method='fir', filter_length='auto', phase='zero', fir_window='hamming')
raw.plot(duration=10.0,start=20.0,n_channels=10,title='EEG Original')
data, times = raw[:, :] 

##Filtrado entre 55 y 65 Hz
#raw=raw.filter(55,65,fir_design='firwin', method='fir', filter_length='auto', phase='zero', fir_window='hamming')
#raw.plot(duration=10.0,start=20.0,n_channels=10,title='EEG Original')
#data, times = raw[:, :] 

#Filtrado nocht
#raw= raw.notch_filter(60)
#data, times = raw[:, :] 


#%%
def ssvepMax (Bipolar,numDataLow,numDataHigh,fs,name):
    #Limpieza de muestras
    deleteL=numDataLow # 250 Eliminar datos iniciales 
    deleteH= len(Bipolar)-numDataHigh # 500 Eliminar datos finales 
    
    Bipolar = Bipolar[deleteL:deleteH]
    
    #periodograma de Welch
    nblock = 2048;
    overlap = nblock/2;
    win = signal.hamming(int(nblock),True);
    
    
    f, Pxx = signal.welch(Bipolar, fs, window=win, noverlap=overlap, nfft=nblock, return_onesided=True);
    plt.plot(f[f<70],Pxx[f<70]);
    plt.ylabel('Amplitud ');
    plt.xlabel('Frecuencia');
    plt.title('Senal');
    plt.grid(True)
    plt.show();
    
    # Estandarización de los datos 
    standarDes = np.std(Pxx, dtype=np.float64)
    media = np.median(Pxx)
    PxxS= (Pxx-media)/standarDes
    plt.figure()
    plt.plot(f[f<30],PxxS[f<30]);
    plt.ylabel('Amplitude');
    plt.xlabel('Frequency (Hz)');
    plt.title('Signal '+ name);
    plt.grid(True)
    plt.show();
    
    
    position = np.where((f >= 9.5) & (f <= 10.5)) # Busca la posición de los valores donde la frecuencia sea 10 o cercana
    maxValue = np.max(PxxS[position[0]])
    
    return maxValue
#%%
fs = 250 # Frecuencia de muestreo
indexMatrix= ['Oz-FCz','PO8-FCz','PO7-FCz','I1+I2-FCz','PO8+P4-FCz','PO8+P3-FCz'] 
signalBipolar = np.zeros((6,totalData))
ssvepValue = np.zeros((6,2))

signalBipolar[0,:] = data[0,:]-data[7,:] # Resta de Oz-FCz
signalBipolar[1,:] = data[6,:]-data[7,:] # Resta de PO8-FCz
signalBipolar[2,:] = data[5,:]-data[7,:] # Resta de PO7-FCz
signalBipolar[3,:] = data[3,:]+data[4,:]-data[7,:] # Resta de I1+I2-FCz
signalBipolar[4,:] = data[6,:]+data[2,:]-data[7,:] # Resta de PO8+P4-FCz
signalBipolar[5,:] = data[5,:]+data[1,:]-data[7,:] # Resta de PO7+P3-FCz

for i in range(0,6):
    ssvepValue[i,0] = ssvepMax(signalBipolar[i,:],250,500,fs,indexMatrix[i])
