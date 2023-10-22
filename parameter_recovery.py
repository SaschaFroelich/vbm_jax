#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:04:09 2023

Parameter Recovery for MCMC with JAX.

@author: sascha
"""
JAX_CHECK_TRACER_LEAKS = True
import ipdb
import numpy as np
import jax
from jax import random as jran
from jax import numpy as jnp
import inference_numpyro as inf
import numpyro

import model_jax as mj
assert(int(jax.__version__.split('.')[1]) >= 4 and int(jax.__version__.split('.')[2]) >= 7)

savedir = '/home/sascha/Desktop/vbm_jax/'

# numpyro.set_platform('gpu')
# jax.default_device=jax.devices("gpu")[0]

numpyro.set_platform('cpu')
jax.default_device=jax.devices("cpu")[0]

level = 2
num_samples = 500
num_warmup = 250
num_agents = 1
if level == 1:
    num_agents = 1

num_sims = 1
Q_init_group = []
num_parameters = 6
k = 4.

key = jran.PRNGKey(np.random.randint(100_000))
# key = jran.PRNGKey(1234)

"Simulate with random parameters"
parameter = jran.uniform(key=key, 
                         minval=0, 
                         maxval=1,
                         shape=(num_sims, num_agents, num_parameters))

_, key = jran.split(key)
lr_day1 = parameter[..., 0]*0.005 # so lr_day1 has shape (1, num_agents)
theta_Q_day1 = parameter[..., 1]*6
theta_rep_day1 = parameter[..., 2]*6

lr_day2 = parameter[..., 3]*0.005
theta_Q_day2 = parameter[..., 4]*6
theta_rep_day2 = parameter[..., 5]*6

errorrates_stt = parameter[..., 7]*0.1
errorrates_dtt = parameter[..., 6]*0.2

sim_parameter = {'lr_day1': lr_day1,
                 'theta_Q_day1' : theta_Q_day1,
                 'theta_rep_day1' : theta_rep_day1,

                 'lr_day2' : lr_day2,
                 'theta_Q_day2' : theta_Q_day2,
                 'theta_rep_day2' : theta_rep_day2,

                 'errorrates_stt' : errorrates_stt,
                 'errorrates_dtt' : errorrates_dtt}

Q_init = jnp.repeat(jnp.asarray([[[0.2, 0., 0., 0.2]]]), 
                    num_agents,
                    axis=1)

Q_init_group.append(Q_init)
print("Simulating data for %d agents."%num_agents)

# print("lr_day1=%.4f"%lr_day1)
# print("lr_day2=%.4f"%lr_day2)

_, _, agent = mj.simulation(num_agents = num_agents,
                           sequence = [1]*num_agents,
                           blockorder = [1]*num_agents,
                            errorrates_stt = errorrates_stt,
                            errorrates_dtt = errorrates_dtt,
                            lr_day1 = lr_day1,
                            lr_day2 = lr_day2,
                            theta_Q_day1 = theta_Q_day1,
                            theta_Q_day2 = theta_Q_day2,
                            theta_rep_day1 = theta_rep_day1,
                            theta_rep_day2 = theta_rep_day2)

if level == 2:
    print("Running group-level inference.")
    mcmc = inf.perform_grouplevelinference(agent = agent, 
                                 num_samples = num_samples,
                                 num_warmup = num_warmup)

    samples = mcmc.get_samples()
    
    locs = samples['locs']
    
    par_dict = agent.locs_to_pars(locs)
        
    means = agent.locs_to_pars(locs.mean(axis=0))
    
    import matplotlib.pyplot as plt
    
    for key in means.keys():
        fig, ax = plt.subplots()
        plt.scatter(np.squeeze(eval(key)), means[key])
        plt.plot(means[key], means[key])
        ax.set_xlabel('True value')
        ax.set_ylabel('Inferred value')
        plt.title(key)
        plt.show()

elif level == 1:
    print("Running first level inference.")
    mcmc = inf.perform_firstlevelinference(agent, num_samples, num_warmup)
    samples = mcmc.get_samples()
    
    import pickle
    file_id = np.random.randint(1e12)
    pickle.dump((samples, sim_parameter), open(savedir + f"param_recov/mcmc_param_recov_{file_id}.p", "wb" ) )

    import contextlib
    with open(savedir + f"param_recov/mcmc_summary_{file_id}.txt", "w" ) as f:
        with contextlib.redirect_stdout(f):
            mcmc.print_summary()
