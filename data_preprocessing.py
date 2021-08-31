import numpy as np
from scipy import signal
from tqdm import tqdm

class DataPreprocessing:
  def __init__(self):
    pass
  def specgram(self, signals, Fs=None, nperseg=None, noverlap=None):
    if Fs==None:
      Fs=360
    if nperseg == None:
      nperseg=64
    if noverlap == None:
      noverlap = int(nperseg/2)
    list_all=[]
    for i in tqdm(range(len(signals))):
      f,t,Sxx= signal.spectrogram(signals[i], fs=Fs, nperseg=nperseg, noverlap=noverlap, mode='psd')
      #arr = Sxx.T[np.newaxis,:,:]
      #print(Sxx.T[:,:].tolist())
      list_all.append(Sxx.T[:,:].tolist())
      #print((list_all))
      '''
      if i == 0:
        out = arr
      else:
        pass
        #out = np.append(out, arr, axis=0)
        out[i,:,:]=arr
      '''
      #i +=1
    out = np.array(list_all)
    return out
  
