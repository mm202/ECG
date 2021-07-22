import numpy as np


def reset_seed(seed_value= 22):
  from numpy import random
  import random as python_random
  import os
  import tensorflow as tf
  #must run every time
  random.seed(seed_value)
  tf.random.set_seed(seed_value)
  python_random.seed(seed_value)
  os.environ['PYTHONHASHSEED']=str(seed_value)


intMapDict = {'N':0,'L':1,'R':2,'V':3,'/':4,'A':5,'F':6,'f':7,'j':8,'a':9,'E':10,'J':11,'e':12,'S':13,'Q':14}

def integerMapping(y,map, inverse=False):
  if not inverse:
    out = [map[i] for i in y]
  else:
    invmap = {v:k for k,v in map.items()}
    out = [invmap[i] for i in y]
  return np.array(out)


MAP_AAMI = {'N':'N', 'L':'N', 'R':'N','j':'N','e':'N', 
             'V':'V', 'E':'V', 
             'A':'S', 'S':'S', 'a':'S',  'J':'S',
             'F':'F',
             'f':'Q', '/':'Q', 'Q':'Q'}


def mappingAAMI(y,map):
  out = [map[i] for i in y]
  return out



def plot_loss(model_history,p=False):
  # Plot training & validation loss
  import matplotlib.pyplot as plt
  if p==True:
    plt.style.use('/content/drive/MyDrive/ecARR/plotstyle.txt')

  plt.plot(model_history.history['loss'])
  plt.plot(model_history.history['val_loss'],'--')
  plt.title('Training Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper right')
  plt.show()

def plot_spectogram(sig):
  import matplotlib.pyplot as plt 
  from scipy import signal

  sampling_rate = 360
  win = 127
  overlap = 122

  #f,t,Sxx= signal.spectrogram(sig, fs=sampling_rate, nperseg=win,nfft=win, noverlap=overlap, mode='psd')
  #Sxx_log = 10*np.log10(Sxx)
  #plt.pcolormesh(t, f, Sxx_log)
  #plt.ylabel('Frequency [Hz]')
  #plt.xlabel('Time [sec]')
  #plt.show()

  import matplotlib.pyplot as plt
  Pxx, freqs, bins, im = plt.specgram(sig, Fs=sampling_rate, NFFT=win, noverlap=overlap, mode='psd')
  print(Pxx.shape)
  expected = (len(sig)-overlap)/(win-overlap)
  print('Expexted:'+str(expected))