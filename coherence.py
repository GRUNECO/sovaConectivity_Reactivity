import scipy.io as sio;
import numpy as np;
import matplotlib.pyplot as plt
import glob
#importamos la rutina de welch
from scipy.signal import coherence
import itertools as it
import pandas as pd

def coherencia(senal,fe):
  nperseg = 
  noverlap = int(nperseg/2)
  fs = 250
  electrodos = senal.shape[1]
  puntos = senal.shape[0]
  #print(senal.shape)
  senal_continua = np.transpose(senal)
  coherencias = []
  for a in range(electrodos):
    for b in range(a,electrodos):
      if a != b:
        fc, Cxyc = coherence(senal_continua[a,:], senal_continua[b,:], fs, 'hanning', nperseg)
        coherencias.append(Cxyc[fe])
  coherencia_promedio = np.mean(coherencias)

  return coherencia_promedio

def coherencia(signal_upload,canales,group,componentes,freq_max,cont=0,cont_fin=None):
  if cont_fin == None:
    cont_fin = len(group) + 1
  nperseg = 500
  noverlap = int(nperseg/2)
  max_frequency = freq_max
  fc_sujetos = np.empty((len(group),len(componentes),len(componentes),1),dtype=object)
  Cxyc_sujetos = np.empty((len(group),len(componentes),len(componentes),1),dtype=object)

  for contador,archivo in enumerate(group[cont:cont_fin]):
    data = signal_upload(archivo,canales)

    puntos,nchannel = data.shape
    senal = data.reshape((nchannel,puntos),order='F')
    sensores = senal.shape[0]
    puntos = senal.shape[1]

    senal_continua = np.reshape(data,(sensores,puntos), order = 'F')

    for num in range(sensores):
      for mun in range(num,sensores):
        if num != mun:
          fc, Cxyc = coherence(senal_continua[num,:], senal_continua[mun,:], fs, 'hanning', nperseg, noverlap)
          fc_sujetos[contador,num,mun] = [fc]
          Cxyc_sujetos[contador,num,mun] = [Cxyc]

  return (fc_sujetos,Cxyc_sujetos)

def grafica_coherencia(conexiones_g1,componentes,sujeto):
  fc_sujetos,Cxyc_sujetos = conexiones_g1
  group = Cxyc_sujetos.shape[0]
  sensores = Cxyc_sujetos.shape[1]
  contador = 0
  for num in range(sensores):
    for mun in range(sensores):
      if num != mun:
        fc = fc_sujetos[sujeto,num,mun][0]
        Cxyc = Cxyc_sujetos[sujeto,num,mun][0]
        if fc is None or Cxyc is None: continue
        fig, ax2 = plt.subplots(1, 1)
        if num != mun:
          ax2.semilogy(fc, Cxyc)
        else:
          ax2.plot(fc, Cxyc)
        ax2.set_title('Coherencia señal continua entre: '+componentes[num]+' y '+componentes[mun]+ ' Sujeto ' + str(sujeto) + ' de ' + str(group))
        ax2.set_xlim([0,50])
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Coherence')
        ax2.grid()

        fig.set_size_inches(20, 5)
        plt.subplots_adjust(wspace=0.5)
        plt.show()
        contador +=1
  print(contador)