import numpy as np
import pandas as pd
import wfdb
import pickle
import os



RECORDS = [100,101,103,105,106,107,108,109,111,
           112,113,114,115,116,117,118,119,121,122,123,
           124,200,201,202,203,205,207,208,209,210,212,
           213,214,215,217,219,220,221,222,223,228,230,
           231,232,233,234]

MAP_AAMI = {'N':'N', 'L':'N', 'R':'N','j':'N','e':'N', 
             'V':'V', 'E':'V', 
             'A':'S', 'S':'S', 'a':'S',  'J':'S',
             'F':'F',
             'f':'Q', '/':'Q', 'Q':'Q'}


class DataHandling:
  def __init__(self, base_path=None, data_path=None, halfWin=None, file_path=None):
    if base_path is None:
      base_path =os.getcwd()
    if data_path is None:
      data_path = "mit-bih-arrhythmia-database-1.0.0/"
    if halfWin is None:
      halfWin = 100
    if file_path is None:
      file_path = 'dataset.dat'

    self.base_path = base_path
    self.halfWin = halfWin
    self.data_path = os.path.join(self.base_path, data_path)
    self.file_path = os.path.join(self.base_path, file_path)
    self.syms = [k for k,v in MAP_AAMI.items()]


  
  def getSignalData(self, data_path, recordNum=106):
    if not data_path:
      data_path = self.data_path
    record = wfdb.rdrecord(data_path + str(recordNum), channel_names=['MLII'])
    annotation = wfdb.rdann(data_path + str(recordNum), 'atr')
    signal = record.p_signal[:,0]
    R_locations = annotation.sample
    R_types = annotation.symbol
    return signal,R_locations,R_types

  def getCuts(self, signal, R_locations=None, R_types=None, halfWin=None):
    if halfWin is None:
      halfWin = self.halfWin
    cuts = []
    typesList = []
    for i in range(len(R_locations)):
      x = signal[R_locations[i]-halfWin:R_locations[i]+halfWin+1]
      if len(x) == 2*halfWin+1:
        cuts.append(x)
        typesList.append(R_types[i])
    signalCuts = np.array(cuts)
    return signalCuts, typesList

  def makeDataset(self, records=None, halfWin=None, data_path=None):
    """ Creates the full dataset
    xds : numpy array of the signal cuts
    yds : list of corresponding types
    """
    if records is None:
      records = RECORDS
    if halfWin is None:
      halfWin = self.halfWin
    if data_path is None:
      data_path = self.data_path
    for i in range(len(records)):
      signal,R_locations,R_types = getSignalData(data_path, recordNum=records[i])
      if i == 0:
        signalCuts, typesList = getCuts(signal, R_locations, R_types, halfWin = halfWin)
        xds = signalCuts
        yds = typesList
      else:  
        signalCuts, typesList = getCuts(signal, R_locations, R_types, halfWin = halfWin)
        xds = np.append(xds,signalCuts,axis=0)
        yds += typesList
      i +=1
    return xds,yds

  def cleanData(self, xds,yds):
    """ Cleans unnecessary symbols"""
    indexes = [i for i,item in enumerate(yds) if item not in self.syms]
    xds = np.delete(xds, indexes, axis=0)
    ydsc = [it for ind,it in enumerate(yds) if ind not in indexes]
    return xds,ydsc

  def searchType(self, xds, yds, Numb, sym='N'):
    """ Search for cutted signal with a patricular type"""
    indexes = [i for i,item in enumerate(yds) if item==sym]
    return xds[indexes[Numb]]

  def reportStats(self, xds, yds):
    """ """
    res = {}
    for sym in self.syms:
      indexes = [i for i,item in enumerate(yds) if item==sym]
      res[sym]=len(indexes)
    return res

  def saveData(self, xds, yds, file_path=None):
    if file_path is None:
      file_path=self.file_path
    with open(file_path, 'wb') as f:
      pickle.dump([xds,yds], f)
    print('File saved in:' + str(file_path))

  def loadData(self, file_path=None):
    if file_path is None:
      file_path=self.file_path
    with open(file_path, 'rb') as f:
      [xds,yds] = pickle.load(f)
    return xds,yds

  def saveDatasetFile(self, clean=True):
    xds,yds = self.makeDataset()
    if clean == True:
      xds,yds = self.cleanData(xds, yds)
    self.saveData(xds,yds)
