#!/usr/bin/env python
# encoding: utf-8
import os
from glob import glob
import pandas as pd
import utils

datapath = "E:/LungCompetition/Lunna"
class Split(object):
    def __init__(self, datapath):
      self.datapath = datapath
      self.all_patients_path = os.path.join(self.datapath, "subset0/")
      self.ls_all_patients = glob(self.all_patients_path + "*.mhd")
      self.df_annotations = pd.read_csv(self.datapath + "/CSVFILES/annotations.csv")
      self.ratio = 0.8
      self.patient_ids = []#从文件名中获取病人的id
      self.train_ids = []
      self.valid_ids = []
      limit = self.ratio * len(self.ls_all_patients)
      idx = 0 
      for p in self.ls_all_patients:
            self.patient_ids.append(os.path.basename(p).split('.')[0])
            if(idx < limit):  
              self.train_ids.append(self.patient_ids[idx])
            else:
              self.valid_ids.append(self.patient_ids[idx])
            idx += 1
      output = {'train':self.train_ids, 'valid':self.valid_ids}
      output_name = datapath + '/validation_split.pkl'
      utils.save_pkl(output, output_name)
       
if __name__ == '__main__':
    s = Split(datapath)


