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

def integerMapping(y,map):
  out = [map[i] for i in y]
  return np.array(out)

def inverseintegerMapping(y,map):
  invmap = {v:k for k,v in map.items()}
  out = [invmap[i] for i in y]
  return out

MAP_AAMI = {'N':'N', 'L':'N', 'R':'N','j':'N','e':'N', 
             'V':'V', 'E':'V', 
             'A':'S', 'S':'S', 'a':'S',  'J':'S',
             'F':'F',
             'f':'Q', '/':'Q', 'Q':'Q'}


def mappingAAMI(y,map):
  out = [map[i] for i in y]
  return out



def plot_loss(model_history):
  # Plot training & validation loss
  import matplotlib.pyplot as plt
  plt.style.use('/content/drive/MyDrive/ecARR/plotstyle.txt')

  plt.plot(model_history.history['loss'])
  plt.plot(model_history.history['val_loss'],'--')
  plt.title('Training Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper right')
  plt.show()