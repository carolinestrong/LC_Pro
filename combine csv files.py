# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:21:49 2021

@author: strongce
"""

#%% MERGE SUMMARY RESULTS FROM ALL WELLS INTO ONE FILE
import os
import glob
import pandas as pd

path = 'C:\LC_Pro\GCaMP\oud model\damgo\Cal6_10-31-21 plate'
os.chdir(path)

all_files = glob.glob(path + "/*.csv")

li = []

for file in all_files:
  df = pd.read_csv(file, index_col=None, header=0)
  li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

frame.to_csv(path + '\\' + 'oud_baseline_summary.csv')
