#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:04:09 2023

Parameter Recovery for MCMC with JAX

@author: sascha
"""
import ipdb
from jax import random as jran
import numpy as np
from jax import numpy as jnp
import model_jax as mj
import inference_numpyro as inf
import numpyro

numpyro.set_platform('cpu')

num_agents = 36
num_sims = 1

Q_init_group = []
npar = 6
k = 4.

key = jran.PRNGKey(np.random.randint(10000))

"Simulate with random parameters"
parameter = jran.uniform(key=key, minval=0, maxval=1,
                         shape=(num_sims, num_agents, npar))

for sim in range(num_sims):
    _, key = jran.split(key)
    lr_day1 = parameter[..., 0]*0.005 # so lr_day1 has shape (1, num_agents)
    theta_Q_day1 = parameter[..., 1]*6
    theta_rep_day1 = parameter[..., 2]*6
    
    lr_day2 = parameter[..., 3]*0.005
    theta_Q_day2 = parameter[..., 4]*6
    theta_rep_day2 = parameter[..., 5]*6
    
    errorrates_stt = parameter[..., 7]*0.1
    errorrates_dtt = parameter[..., 6]*0.2
    
    Q_init = jnp.repeat(jnp.asarray([[[0.2, 0., 0., 0.2]]]), 
                        num_agents,
                        axis=1)
    
    Q_init_group.append(Q_init)
    
    print("Simulating data for %d agents"%num_agents)
    _, exp_data, agent = mj.simulation(num_agents = num_agents,
                                errorrates_stt = jnp.ones((1,num_agents))*0.1,
                                errorrates_dtt = jnp.ones((1,num_agents))*0.2,
                                lr_day1 = lr_day1,
                                lr_day2 = lr_day2,
                                theta_Q_day1 = theta_Q_day1,
                                theta_Q_day2 = theta_Q_day2,
                                theta_rep_day1 = theta_rep_day1,
                                theta_rep_day2 = theta_rep_day2)
    
    print("Running inference")
    mcmc = inf.perform_inference(agent = agent,
                                 exp_data = exp_data)
    samples = mcmc.get_samples()

#%%

