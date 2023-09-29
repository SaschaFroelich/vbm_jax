#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:26:59 2023

Simulation and parallel group-level inference

@author: sascha
"""

# import sys
# sys.modules[__name__].__dict__.clear()

import ipdb

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import torch

import numpy
import scipy

import models_torch as models

import env
import utils

import inferencemodels
# import inferencemodel_sarah as inferencemodels

import sys
from datetime import datetime
import pickle

plt.style.use("classic")

#%%
# model = sys.argv[1]
# resim =  int(sys.argv[2]) # whether to simulate agents with inferred parameters
# method = sys.argv[3] # "svi" or "mcmc"
# num_agents = 50

model = 'B'
resim =  0 # whether to simulate agents with inferred parameters
method = 'svi' # "svi" or "mcmc"
num_agents = 2
k = 4.
print(f"Running model {model}")

#%%

if resim:
    raise Exception("Not implemented yet, buddy!")
    
agents = []
groupdata = []
Q_init_group = []    


if model == 'original':
    npar = 3
    
    omega_true = []
    dectemp_true = []
    lr_true = []
    
    "Simulate with random parameters"
    for ag in range(num_agents):
        print("Simulating agent no. %d"%ag)
        parameter = numpy.random.uniform(0,1, npar)
        omega = parameter[0]
        dectemp = (parameter[1]+1)*3
        lr = parameter[2]*0.01
        
        omega_true.append(omega)
        dectemp_true.append(dectemp)
        lr_true.append(lr)
        
        Q_init = [0.2, 0., 0., 0.2]
        Q_init_group.append(Q_init)
        newagent = models.vbm(omega = torch.tensor([[omega]]),
                              dectemp = torch.tensor([[dectemp]]),
                              lr = torch.tensor([[lr]]),
                              k=torch.tensor(k),
                              Q_init=torch.tensor([[Q_init]]))
            
        newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir = './matlabcode/clipre/')
        
        newenv.run()
        
        data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,
                "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],
                    "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}
            
        # utils.plot_results(pd.DataFrame(data), group = 0)
            
        groupdata.append(data)
        
elif model == 'vbm_timedep':
    npar = 4
    
    omega_true = []
    dectemp0_true = []
    alpha_true = []
    lr_true = []
    
    "Simulate with random parameters"
    for ag in range(num_agents):
        print("Simulating agent no. %d"%ag)
        parameter = numpy.random.uniform(0,1, npar)
        omega = parameter[0]
        dectemp0 = parameter[1]*2
        alpha = parameter[2]*3
        lr = parameter[3]*0.2
        
        omega_true.append(omega)
        dectemp0_true.append(dectemp0)
        alpha_true.append(alpha)
        lr_true.append(lr)
        
        Q_init = [0.2, 0., 0., 0.2]
        Q_init_group.append(Q_init)
        newagent = models.vbm_timedep(omega = torch.tensor([[omega]]), \
                              dectemp0 = torch.tensor([[dectemp0]]), \
                              alpha = torch.tensor([[alpha]]), \
                              lr = torch.tensor([[lr]]), \
                              k=torch.tensor(k),\
                              Q_init=torch.tensor([[Q_init]]))
            
        newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir = './matlabcode/clipre/')
        
        newenv.run()
        
        data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
                "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
                    "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}
            
        # utils.plot_results(pd.DataFrame(data), group = 0)
            
        groupdata.append(data)
   
elif model == 'vbm_timedep2':
    npar = 3
    
    omega_true = []
    dectemp0_true = []
    alpha_true = []
    
    "Simulate with random parameters"
    for ag in range(num_agents):
        print("Simulating agent no. %d"%ag)
        parameter = numpy.random.uniform(0,1, npar)
        omega = parameter[0]
        dectemp0 = parameter[1]*2
        alpha = parameter[2]*3
        
        omega_true.append(omega)
        dectemp0_true.append(dectemp0)
        alpha_true.append(alpha)
        
        Q_init = [0.8, 0.2, 0.2, 0.8]
        Q_init_group.append(Q_init)
        newagent = models.vbm_timedep2(omega = torch.tensor([[omega]]), \
                              dectemp0 = torch.tensor([[dectemp0]]), \
                              alpha = torch.tensor([[alpha]]), \
                              k=torch.tensor(k),\
                              Q_init=torch.tensor([[Q_init]]))

        newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir = './matlabcode/clipre/')
        
        newenv.run()
        
        data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
                "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
                "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}
            
        # utils.plot_results(pd.DataFrame(data), group = 0)
            
        groupdata.append(data)
   
   
elif model == 'B':
    npar = 6
    
    lr_day1_true = []
    theta_Q_day1_true = []
    theta_rep_day1_true = []
    
    lr_day2_true = []
    theta_Q_day2_true = []
    theta_rep_day2_true = []
        
    "Simulate with random parameters"
    for ag in range(num_agents):
        print("Simulating agent no. %d"%ag)
        parameter = numpy.random.uniform(0,1, npar)
        lr_day1 = parameter[0]*0.01
        theta_Q_day1 = parameter[1]*6
        theta_rep_day1 = parameter[2]*6
        
        lr_day2 = parameter[3]*0.01
        theta_Q_day2 = parameter[4]*6
        theta_rep_day2 = parameter[5]*6
        
        lr_day1_true.append(lr_day1)
        theta_Q_day1_true.append(theta_Q_day1)
        theta_rep_day1_true.append(theta_rep_day1)
        
        lr_day2_true.append(lr_day2)
        theta_Q_day2_true.append(theta_Q_day2)
        theta_rep_day2_true.append(theta_rep_day2)

        Q_init = [0.2, 0., 0., 0.2]
        Q_init_group.append(Q_init)
        newagent = models.vbm_B(lr_day1 = torch.tensor([[lr_day1]]), \
                              theta_Q_day1 = torch.tensor([[theta_Q_day1]]), \
                              theta_rep_day1 = torch.tensor([[theta_rep_day1]]), \
                                  
                              lr_day2 = torch.tensor([[lr_day2]]), \
                              theta_Q_day2 = torch.tensor([[theta_Q_day2]]), \
                              theta_rep_day2 = torch.tensor([[theta_rep_day2]]), \
                              k=torch.tensor(k),\
                              Q_init=torch.tensor([[Q_init]]))
            
        newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir = './matlabcode/clipre/')
        
        newenv.run()
        
        data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
                "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
                    "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}
            
        # utils.plot_results(pd.DataFrame(data), group = 0)
            
        groupdata.append(data)  
        
elif model == 'Bpersbias1':
    npar = 8
    
    lr_day1_true = []
    theta_Q_day1_true = []
    theta_rep_day1_true = []
    b_day1_true = []
    
    lr_day2_true = []
    theta_Q_day2_true = []
    theta_rep_day2_true = []
    b_day2_true = []
        
    "Simulate with random parameters"
    for ag in range(num_agents):
        print("Simulating agent no. %d"%ag)
        parameter = numpy.random.uniform(0,1, npar)
        lr_day1 = parameter[0]*0.1
        theta_Q_day1 = parameter[1]*6
        theta_rep_day1 = parameter[2]*6
        b_day1 = parameter[3]*6
        
        lr_day2 = parameter[4]*0.1
        theta_Q_day2 = parameter[5]*6
        theta_rep_day2 = parameter[6]*6
        b_day2 = parameter[7]*6
        
        lr_day1_true.append(lr_day1)
        theta_Q_day1_true.append(theta_Q_day1)
        theta_rep_day1_true.append(theta_rep_day1)
        b_day1_true.append(b_day1)
        
        lr_day2_true.append(lr_day2)
        theta_Q_day2_true.append(theta_Q_day2)
        theta_rep_day2_true.append(theta_rep_day2)
        b_day2_true.append(b_day2)

        Q_init = [0.2, 0., 0., 0.2]
        Q_init_group.append(Q_init)
        newagent = models.vbm_Bpersbias1(lr_day1 = torch.tensor([[lr_day1]]), \
                              theta_Q_day1 = torch.tensor([[theta_Q_day1]]), \
                              theta_rep_day1 = torch.tensor([[theta_rep_day1]]), \
                              b_day1 = torch.tensor([[b_day1]]), \
                                  
                              lr_day2 = torch.tensor([[lr_day2]]), \
                              theta_Q_day2 = torch.tensor([[theta_Q_day2]]), \
                              theta_rep_day2 = torch.tensor([[theta_rep_day2]]), \
                              b_day2 = torch.tensor([[b_day2]]), \
                                  
                              k=torch.tensor(k),\
                              Q_init=torch.tensor([[Q_init]]))
            
        newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir = './matlabcode/clipre/')
        
        newenv.run()
        
        data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
                "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
                    "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}
            
        # utils.plot_results(pd.DataFrame(data), group = 0)
            
        groupdata.append(data)  

elif model == 'testmodel':
    npar = 2
    
    agents = []
    groupdata = []
    
    prob1_true = []
    prob2_true = []
    
    "Simulate with random parameters"
    for ag in range(num_agents):
        print("Simulating agent no. %d"%ag)
        parameter = numpy.random.uniform(0,1, npar)
        prob1 = parameter[0]
        prob2 = parameter[1]
        
        prob1_true.append(prob1)
        prob2_true.append(prob2)

        newagent = models.testmodel(prob1 = torch.tensor([[prob1]]), prob2 = torch.tensor([[prob2]]))
            
        newenv = env.env(newagent, rewprobs=[0.8, 0.2, 0.2, 0.8], matfile_dir = './matlabcode/clipre/')
        
        newenv.run()
        
        data = {"Choices": newenv.choices, "Outcomes": newenv.outcomes,\
                "Trialsequence": newenv.data["trialsequence"], "Blocktype": newenv.data["blocktype"],\
                    "Jokertypes": newenv.data["jokertypes"], "Blockidx": newenv.data["blockidx"]}
            
        # utils.plot_results(pd.DataFrame(data), group = 0)
            
        groupdata.append(data)  

newgroupdata = utils.comp_groupdata(groupdata, for_ddm = 0)

if model == 'original':
    agent = models.vbm(omega = torch.tensor([omega_true]), \
                       dectemp = torch.tensor([dectemp_true]), \
                       lr = torch.tensor([lr_true]), \
                       k = torch.tensor(k), \
                       Q_init = torch.tensor([Q_init_group]))

elif model == 'vbm_timedep':
    agent = models.vbm_timedep(omega = torch.tensor([omega_true]), \
                       dectemp0 = torch.tensor([dectemp0_true]), \
                       alpha = torch.tensor([alpha_true]), \
                       lr = torch.tensor([lr_true]), \
                       k = torch.tensor(k), \
                       Q_init = torch.tensor([Q_init_group]))
        
elif model == 'vbm_timedep2':
    agent = models.vbm_timedep2(omega = torch.tensor([omega_true]), \
                       dectemp0 = torch.tensor([dectemp0_true]), \
                       alpha = torch.tensor([alpha_true]), \
                       k = torch.tensor(k), \
                       Q_init = torch.tensor([Q_init_group]))

elif model == 'B':
    agent = models.vbm_B(lr_day1 = torch.tensor([lr_day1_true]), \
                          theta_Q_day1 = torch.tensor([theta_Q_day1_true]), \
                          theta_rep_day1 = torch.tensor([theta_rep_day1_true]), \

                          lr_day2 = torch.tensor([lr_day2_true]), \
                          theta_Q_day2 = torch.tensor([theta_Q_day2_true]), \
                          theta_rep_day2 = torch.tensor([theta_rep_day2_true]), \
                          k = torch.tensor(k),\
                          Q_init = torch.tensor([Q_init_group]))

elif model == 'Bpersbias1':
    agent = models.vbm_Bpersbias1(lr_day1 = torch.tensor([lr_day1_true]), \
                          theta_Q_day1 = torch.tensor([theta_Q_day1_true]), \
                          theta_rep_day1 = torch.tensor([theta_rep_day1_true]), \
                          b_day1 = torch.tensor([b_day1_true]), \

                          lr_day2 = torch.tensor([lr_day2_true]), \
                          theta_Q_day2 = torch.tensor([theta_Q_day2_true]), \
                          theta_rep_day2 = torch.tensor([theta_rep_day2_true]), \
                          b_day2 = torch.tensor([b_day2_true]), \
                              
                          k=torch.tensor(k),\
                          Q_init=torch.tensor([Q_init_group]))

elif model == 'testmodel':
    agent = models.testmodel(prob1 = torch.tensor([prob1_true]), prob2 = torch.tensor([prob2_true]))

print("===== Starting inference =====")
infer = inferencemodels.GeneralGroupInference(agent, num_agents, newgroupdata)
infer.infer_posterior(iter_steps = 2500, num_particles = 20)
inference_df = infer.sample_posterior()
df = inference_df.groupby(['subject']).mean()

for key in agent.param_names:
    df[key + "_true"] = globals()[key + "_true"]

"Plot ELBO"
plt.plot(infer.loss)
plt.show()

print("Saving data files!")
pickle.dump( inference_df, open(f"param_recov/groupinference/model{model}/k_{k}/group_inference_samples.p", "wb" ) )
pickle.dump( df, open(f"param_recov/groupinference/model{model}/k_{k}/group_inference_means.p", "wb" ) )
pickle.dump( infer.loss, open(f"param_recov/groupinference/model{model}/k_{k}/ELBO_group_inference.p", "wb" ) )

#%%
"Load results from local files"
if 1:
    import os 
    model = 'vbm_timedep2'
    datadir = f'param_recov/groupinference/model{model}/k_{k}/'
    version = 1
    
    if model == 'original':
        param_names = ['omega', 'dectemp', 'lr']
        
    elif model == 'vbm_timedep':
        if version < 3:
            param_names = ['omega', 'dectemp0', 'dectemp1', 'lr']
        elif version >= 3:
            param_names = ['omega', 'dectemp0', 'alpha', 'lr']
            
    elif model == 'vbm_timedep2':
            param_names = ['omega', 'dectemp0', 'alpha']
    
    "Prior"
    # prior_df = pickle.load( open(os.fsencode(datadir +  f'version{version}_' +  "prior.p"), "rb" ) )
    
    # df = pd.DataFrame()
    
    # for file in os.listdir(datadir):
    #      filename = os.fsdecode(file)
    #      if filename.endswith(".p") and "ELBO" not in filename and "prior" not in filename and f'version{version}' in filename:
    #          # print("opening %s"%(directory + file))
             
    #          pickle_df = pickle.load( open(datadir + file, "rb" ) )
                      
    #          df = pd.concat((df, pickle_df))
    
    "ELBO"
    fig, ax = plt.subplots()
    for file in os.listdir(datadir):
          filename = os.fsdecode(file)
          if filename.endswith(".p") and "ELBO" in filename and f"version{version}" in filename:
            loss = pickle.load( open(datadir +  file, "rb" ) )
            
            ax.plot(loss)
            plt.plot([1000,1000],[15000, 21000], color='k')
            plt.title("ELBO")
            ax.set_ylabel("ELBO")
            ax.set_xlabel("iteration steps")
            # plt.savefig(datadir + f"version{version}_ELBO.png")
            plt.show()
             
    # print("========================================================")
    # for col in prior_df.columns:
    #     if col == 'Q_init':
    #         print("%s prior = "%col)
    #         print(prior_df[col][0])
    #     else:
    #         print("%s prior = %d"%(col, prior_df[col][0]))
    
    df = pickle.load( open(datadir +  f"group_inference_means_version{version}.p", "rb" ) )
    
    import numpy as np
    
    "Plot inference means"
    for key in param_names:
        plt.scatter(df[key + "_true"], df[key])
        plt.plot(df[key + "_true"], df[key + "_true"])
        plt.title(key, size = 20)
        plt.xlabel("True (simulated) value", size = 20)
        plt.ylabel("Inferred value", size=20)
        plt.show()
    
    #%%
    
    
    "Plot distributions"
    for ag in range(num_agents):
        for par in inference_df.columns:
            if par != 'subject':
                sns.kdeplot(inference_df[inference_df["subject"]==ag][par])
                plt.title("agent %d"%ag)
                plt.show()


#%%

