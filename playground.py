#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:13:46 2023

@author: sascha
"""

import model_jax as mj
from jax import numpy as jnp

num_agents = 1

theta_rep_day1 = 0.8
theta_rep_day2 = 0.8
theta_Q_day1 = 3.
theta_Q_day2 = 3.
lr_day1 = 0.005
lr_day2 = 0.

sim_df = mj.simulation(num_agents = num_agents, 
                       lr_day1 = jnp.asarray([[lr_day1]*num_agents]),
                       lr_day2 = jnp.asarray([[lr_day2]*num_agents]), 
                       
                       theta_Q_day1 = jnp.asarray([[theta_Q_day1]*num_agents]),
                       theta_Q_day2 = jnp.asarray([[theta_Q_day2]*num_agents]),
                       
                       theta_rep_day1 = jnp.asarray([[theta_rep_day1]*num_agents]),
                       theta_rep_day2 = jnp.asarray([[theta_rep_day2]*num_agents]))

'''
Jokertypes:
    -1 no joker
    0 random
    1 congruent
    2 incongruent
'''
mj.plot_simulated(sim_df)