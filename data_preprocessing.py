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
    for i in tqdm(range(len(signals))):
      f,t,Sxx= signal.spectrogram(signals[i], fs=Fs, nperseg=nperseg, noverlap=noverlap, mode='psd')
      arr = Sxx.T[np.newaxis,:,:]
      if i == 0:
        out = arr
      else:
        out = np.append(out, arr, axis=0)
      i +=1
    return out
  