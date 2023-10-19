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
from jax import vmap
import inference_numpyro as inf
import numpyro

import model_jax as mj

assert(int(jax.__version__.split('.')[1]) >= 4 and int(jax.__version__.split('.')[2]) >= 7)

numpyro.set_platform('gpu')
jax.default_device=jax.devices("gpu")[0]

num_agents = 6
level = 2
num_sims = 1
num_samples = 500
num_warmup = 1000

Q_init_group = []
num_parameters = 6
k = 4.

key = jran.PRNGKey(np.random.randint(10_000))

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

Q_init = jnp.repeat(jnp.asarray([[[0.2, 0., 0., 0.2]]]), 
                    num_agents,
                    axis=1)

Q_init_group.append(Q_init)
print("Simulating data for %d agents."%num_agents)
_, exp_data, agent = mj.simulation(num_agents = num_agents,
                           sequence = [1]*num_agents,
                           blockorder = [1]*num_agents,
                            errorrates_stt = jnp.ones((1,num_agents))*0.1,
                            errorrates_dtt = jnp.ones((1,num_agents))*0.2,
                            lr_day1 = lr_day1,
                            lr_day2 = lr_day2,
                            theta_Q_day1 = theta_Q_day1,
                            theta_Q_day2 = theta_Q_day2,
                            theta_rep_day1 = theta_rep_day1,
                            theta_rep_day2 = theta_rep_day2)

print("Running inference.")
mcmc = inf.perform_grouplevel_inference(agent = agent, 
                             num_samples = num_samples,
                             num_warmup = num_warmup,
                             level = level)

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
    

