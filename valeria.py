import numpy as np


#Se inicializan las listas
f_n=[]
px_n=[]
media_n=[]

#Se halla el PSD para ciclos normales
for i in range(0,len(datos)):
  #Estimación de la densidad espectral con una ventana tipo hamming. 
  fn, Pn = signal.welch(d_filtrado[i],fm, 'hamming',2*fm, scaling='density')
  px_n.append(Pn)
  f_n.append(fn)

lis_frec=[]
for i in range(len(d_filtrado)):
  max_amp=np.max(px_n[i])
  
  posicion=np.where(px_n[i]==max_amp)[0][0]
  frec=f_n[i][posicion]
  lis_frec.append(frec)
  print('Amplitud maxima: ',max_amp)
  print('Frecuencia: ',frec)
  plt.figure(figsize=(15,5))
  plt.subplot(1,2,1)
  plt.title('PSD')
  plt.xlabel('Frecuencia [Hz]')
  plt.ylabel('Amplitud')
  plt.grid()
  plt.plot(f_n[i], px_n[i])
  plt.subplot(1,2,2)
  plt.title('PSD')
  plt.xlabel('Frecuencia [Hz]')
  plt.ylabel('Amplitud')
  plt.xlim(0,100)
  plt.grid()
  plt.plot(f_n[i], px_n[i])
  plt.show()