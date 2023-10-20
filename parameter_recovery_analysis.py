#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:43:22 2023

For analysis of parameter recovery

@author: sascha
"""

import numpy as np
import pandas as pd
import pickle
import os

directory = '/home/sascha/Desktop/vbm_jax/param_recov/'

all_samples = []
all_parameters = []

for file in os.listdir(os.fsencode(directory)):
     filename = os.fsdecode(file)
     if filename.endswith(".p"):
         # print("opening %s"%(os.fsencode(directory) + file))
         
         samples, sim_parameter = pickle.load( open(os.fsencode(directory) + file, "rb" ) )
         
         all_samples.append(samples)
         all_parameters.append(sim_parameter)

print("Number of samples is %d"%np.size(samples['lr_day1']))
param_recov_df = pd.DataFrame(columns=["agent", "parameter", "true_value", "est_mean"])
agent = []
parameter = []
true_value = []
est_mean = []

for ag in range(len(all_samples)):
    
    for key in all_samples[ag].keys():
        agent.append(ag)
        parameter.append(key)
        true_value.append(all_parameters[ag][key].item())
        est_mean.append(all_samples[ag][key].mean().item())
        
for col in param_recov_df.columns:
    param_recov_df[col] = eval(col)

import matplotlib.pyplot as plt    
import seaborn as sns

for key in all_parameters[0].keys():
    print(key)
    if key != 'errorrates_stt' and key != 'errorrates_dtt':
        fig, ax = plt.subplots()
        plot = sns.scatterplot(x='true_value', y='est_mean', data = param_recov_df[param_recov_df["parameter"]==key])
        plt.plot(param_recov_df[param_recov_df["parameter"]==key]['true_value'], param_recov_df[param_recov_df["parameter"]==key]['true_value'])
        plt.title(key)
        plt.plot()
        ax.set_xlabel('true value')
        ax.set_ylabel('est mean value')
        plt.show()