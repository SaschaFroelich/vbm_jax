#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:52:23 2023

@author: sascha
"""
#%%

"""
Test jax.vmap()
Comparison within same file
Compare outputs of new vmapped update() with old update
"""

import jax
from jax import numpy as jnp

num_agents = 3
blocktype = jnp.asarray([0,0,1])
pppchoice = jnp.asarray([0,2,1])
ppchoice = jnp.asarray([0,3,1])
pchoice = jnp.asarray([2,3,1])
choice = jnp.asarray([0,3,1])

seq_counter = 4.0 / 4 * jnp.ones((num_agents, 2, 6, 6, 6, 6))

def update_habitual(seq_counter_agent, blocktype, ppp, pp, p, c):
    indices = jnp.indices(seq_counter_agent.shape)
    pos = (blocktype, ppp, pp, p, c)
    
    " Update counter "
    "%6 such that -1 & -2 get mapped to 5 and 4"
    seq_counter_agent = jnp.where((indices[0] == pos[0]%6) & 
                        (indices[1] == pos[1]%6) & 
                        (indices[2] == pos[2]%6) &
                        (indices[3] == pos[3]%6) &
                        (indices[4] == pos[4]%6), seq_counter_agent+1, seq_counter_agent)
    
    "Update rep values"
    index = (blocktype, pp, p, c)
    # dfgh
    seqs_sum = seq_counter_agent[index + (0,)] + seq_counter_agent[index + (1,)] + \
                seq_counter_agent[index + (2,)] + seq_counter_agent[index + (3,)]
    
    new_row_agent = jnp.asarray([seq_counter_agent[index + (aa,)] / seqs_sum for aa in range(4)])
    
    return seq_counter_agent, new_row_agent


seq_counter, jrepnew = jax.vmap(update_habitual, in_axes = (0,0,0,0,0,0))(seq_counter, 
                                                       blocktype, 
                                                       pppchoice, 
                                                       ppchoice, 
                                                       pchoice, 
                                                       choice)

"-----"

import model_jax as mj

from jax import random as jran
import jax
from jax import numpy as jnp

num_agents = 10

Q_init_group = []
num_parameters = 6
k = 4.

# key = jran.PRNGKey(900)
key = jran.PRNGKey(np.random.randint(10000))

"Simulate with random parameters"
"Simulate with random parameters"
parameter = jran.uniform(key=key, minval=0, maxval=1,
                         shape=(1, num_agents, 6))

_, key = jran.split(key)
lr_day1 = parameter[..., 0]*0.01 # so lr_day1 has shape (1, num_agents)
theta_Q_day1 = parameter[..., 1]*6
theta_rep_day1 = parameter[..., 2]*6

lr_day2 = parameter[..., 3]*0.01
theta_Q_day2 = parameter[..., 4]*6
theta_rep_day2 = parameter[..., 5]*6

Q_init = jnp.repeat(jnp.asarray([[[0.2, 0., 0., 0.2]]]), num_agents,
                    axis=1)
Q_init_group.append(Q_init)

agent = mj.Vbm_B(lr_day1=jnp.asarray(lr_day1),
              theta_Q_day1=jnp.asarray(theta_Q_day1),
              theta_rep_day1=jnp.asarray(theta_rep_day1),
              lr_day2=jnp.asarray(lr_day2),
              theta_Q_day2=jnp.asarray(theta_Q_day2),
              theta_rep_day2=jnp.asarray(theta_rep_day2),
              k=k,
              Q_init=jnp.asarray(Q_init))


# for key in exp_data.keys():
#     print(key)
#     assert(np.all(exp_data_new[key]==exp_data_old[key]))
    
# len(np.where(exp_data_new['choices']!=exp_data_old['choices'])[0])


import numpy as np
num_agents = 10


trial = np.array([-1]*num_agents)
choices = np.array([-1]*num_agents)

trial = np.array([12,12,12,12,12,34,34,34,34,34])
choices = np.array([-2, 0, 0, 0, 1, 2, 2, -2, 3, 2])

trial = np.array([1,2,1,2,1,3,4,3,4,3])
choices = np.array([0,1,-2,1,0,2,3,2,3,-2])


outcome  = np.random.randint(0,2,size=num_agents)
blocktype = np.random.randint(0,2,size=num_agents)
day = np.random.randint(0,2,size=num_agents)
Q = np.random.rand(1, num_agents, 4)
pppchoice = np.random.randint(0,4,size=num_agents)
ppchoice  = np.random.randint(0,4,size=num_agents) 
pchoice = np.array([0,1,0,1,0,2,3,2,3,-2])
seq_counter =  jnp.asarray(np.random.randint(0,10_000,size=(num_agents,2,6,6,6,6)))
rep = [jnp.ones((1, num_agents, 4))/4]
V = [np.random.rand(1,num_agents, 4)]
lr_day1 = np.random.rand(1,10)
lr_day2 = np.random.rand(1,10)
theta_Q_day1 = np.random.rand(1,10)*6
theta_Q_day2 = np.random.rand(1,10)*6
theta_rep_day1 = np.random.rand(1,10)*6
theta_rep_day2 = np.random.rand(1,10)*6

out=agent.update(choices, 
               outcome, 
               blocktype, 
               day, 
               trial, 
               Q, 
               pppchoice, 
               ppchoice, 
               pchoice,
               seq_counter,
               rep,
               V,
               lr_day1,
               lr_day2,
               theta_Q_day1,
               theta_Q_day2,
               theta_rep_day1,
               theta_rep_day2)


"----"
"Compare outputs of new vmapped update() with old update"
import model_jax as mj

from jax import random as jran
import jax
from jax import numpy as jnp

# del exp_data

num_agents = 2
level = 2
num_sims = 1
num_samples = 50
num_warmup = 20

Q_init_group = []
num_parameters = 6
k = 4.

key = jran.PRNGKey(900)
# key = jran.PRNGKey(np.random.randint(10000))

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
_, exp_data_new, agent = mj.simulation(num_agents = num_agents,
                                   key = key,
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


#%%
'''
Test new Qoutcomp
'''
from jax import numpy as jnp
import numpy as np
num_agents = 3
BAD_CHOICE = -2
NA = 4

Qin = jnp.asarray(np.random.rand(1, num_agents,4))
choices = jnp.asarray(np.random.choice([-2,0,1,2,3], size = num_agents))

"----Old Code"
"np_error_mask will be True for those subjects that performed no error"         
no_error_mask = jnp.array(choices) != BAD_CHOICE
"Replace error choices by the number one"
choices_noerrors = jnp.where(jnp.asarray(no_error_mask, dtype=bool),
                             choices, jnp.ones(choices.shape)).astype(int)


choicemask = jnp.zeros(Qin.shape, dtype=int)
num_particles = Qin.shape[0]  # num of particles
# assert(num_particles==1)
num_agents = Qin.shape[1]  # num_agents

"Old Code"
no_error_mask = jnp.broadcast_to(
    no_error_mask, (num_particles, 4, num_agents)).transpose(0, 2, 1)
choicemask = jnp.eye(NA)[choices_noerrors, ...].astype(int)
maskold = no_error_mask*choicemask

"----New Code"
"np_error_mask will be True for those subjects that performed no error"         
no_error_mask = jnp.array(choices) != BAD_CHOICE
"Replace error choices by the number one"
choices_noerrors = jnp.where(jnp.asarray(no_error_mask, dtype=bool),
                         choices, jnp.ones(choices.shape)).astype(int)


choicemask = jnp.zeros(Qin.shape, dtype=int)
num_particles = Qin.shape[0]  # num of particles
# assert(num_particles==1)
num_agents = Qin.shape[1]  # num_agents

"Old Code"
# no_error_mask = jnp.broadcast_to(
#     no_error_mask, (num_particles, 4, num_agents)).transpose(0, 2, 1)
# choicemask = jnp.eye(self.NA)[choices_noerrors, ...].astype(int)
# mask = no_error_mask*choicemask

# dfgh
"New Code"
choicemask = jnp.eye(NA)[choices_noerrors, ...].astype(int)
masknew = no_error_mask[None,...,None]*choicemask

print(Qin)
print(choices)
assert(jnp.all(masknew==maskold))

#%%
'''
In compute_probs
'''
from itertools import product
from jax import numpy as jnp
num_particles = 1
num_agents = 12

import time 
start = time.time()

ind_all_old = jnp.array(list(product(jnp.arange(num_particles),
                                  jnp.arange(num_agents))))
print("Finished old code in %.6f seconds"%(time.time()-start))

import time 
start = time.time()
ind_all_new=jnp.transpose(jnp.stack((jnp.zeros(num_agents),jnp.arange(num_agents)))).astype(int)
print("Finished new code in %.6f seconds"%(time.time()-start))

# print(ind_all)

assert(jnp.all(ind_all_new==ind_all_old))

#%%
'''
Test new part in update()
'''
"Old Code"
import utils 
from jax import numpy as jnp
import numpy as np
num_agents = 10

choices = jnp.asarray(np.random.choice([-2,0,1,2,3], size = num_agents))
trial = jnp.asarray(np.random.choice([0,1,2,3,12,13,14,23,24,34], size = num_agents))

outcome  = np.random.randint(0,2,size=num_agents)
blocktype = np.random.randint(0,2,size=num_agents)
day = np.random.randint(0,2,size=num_agents)
Q = np.random.rand(1, num_agents, 4)
pppchoice = np.random.randint(0,4,size=num_agents)
ppchoice  = np.random.randint(0,4,size=num_agents) 
pchoice = np.array([0,1,0,1,0,2,3,2,3,-2])
seq_counter =  jnp.asarray(np.random.randint(0,10_000,size=(num_agents,2,6,6,6,6)))
rep = [jnp.ones((1, num_agents, 4))/4]
V = [np.random.rand(1,num_agents, 4)]
lr_day1 = np.random.rand(1,10)
lr_day2 = np.random.rand(1,10)
theta_Q_day1 = np.random.rand(1,10)*6
theta_Q_day2 = np.random.rand(1,10)*6
theta_rep_day1 = np.random.rand(1,10)*6
theta_rep_day2 = np.random.rand(1,10)*6

agent = utils.init_random_agent(num_agents)

out=agent.update(choices, 
               outcome, 
               blocktype, 
               day, 
               trial, 
               Q, 
               pppchoice, 
               ppchoice, 
               pchoice,
               seq_counter,
               rep,
               V,
               lr_day1,
               lr_day2,
               theta_Q_day1,
               theta_Q_day2,
               theta_rep_day1,
               theta_rep_day2)


#%%
from jax import numpy as jnp
"test alternative to vmap"
num_agents = 3

seq_counter = 4.0 / 4 * jnp.ones((num_agents, 2, 6, 6, 6, 6))

blocktype = jnp.asarray([0,0,1])
pppchoice = jnp.asarray([0,2,1])
ppchoice = jnp.asarray([0,3,1])
pchoice = jnp.asarray([2,3,1])
choice = jnp.asarray([0,3,1])

indices = jnp.indices(seq_counter.shape)

mask = (indices[0] == jnp.arange(num_agents)[:,None,None,None,None,None]) & \
    (indices[1] == blocktype) & (indices[2] == pppchoice) \
    & (indices[3] == ppchoice) & (indices[3] == pchoice) & (indices[3] == choice)
    
seq_counter_new = jnp.where()

#%%

x,y,z = [0,1,2], [0,0,1], [0,3,2]

jnp.broadcast_to(jnp.eye(4)[z,...][:,None], (3,2,4))

#%%
"Test Qoutcomp"
import numpy as np
import model_jax as mj

from jax import random as jran
import jax
from jax import numpy as jnp

num_agents = 10

Q_init_group = []
num_parameters = 6
k = 4.

# key = jran.PRNGKey(900)
key = jran.PRNGKey(np.random.randint(10000))

"Simulate with random parameters"
"Simulate with random parameters"
parameter = jran.uniform(key=key, minval=0, maxval=1,
                         shape=(1, num_agents, 6))

_, key = jran.split(key)
lr_day1 = parameter[..., 0]*0.01 # so lr_day1 has shape (1, num_agents)
theta_Q_day1 = parameter[..., 1]*6
theta_rep_day1 = parameter[..., 2]*6

lr_day2 = parameter[..., 3]*0.01
theta_Q_day2 = parameter[..., 4]*6
theta_rep_day2 = parameter[..., 5]*6

Q_init = jnp.repeat(jnp.asarray([[[0.2, 0., 0., 0.2]]]), num_agents,
                    axis=1)
Q_init_group.append(Q_init)

agent = mj.Vbm_B(lr_day1=jnp.asarray(lr_day1),
              theta_Q_day1=jnp.asarray(theta_Q_day1),
              theta_rep_day1=jnp.asarray(theta_rep_day1),
              lr_day2=jnp.asarray(lr_day2),
              theta_Q_day2=jnp.asarray(theta_Q_day2),
              theta_rep_day2=jnp.asarray(theta_rep_day2),
            errorrates_stt = jnp.asarray(np.random.rand(1, num_agents))*0.1,
            errorrates_dtt = jnp.asarray(np.random.rand(1, num_agents))*0.2,
              k=k,
              Q_init=jnp.asarray(Q_init))

# choices = np.array([0,1,-2,1,0,2,3,2,3,-2])
choices = jnp.asarray(np.random.choice([-2, -1, 0, 1, 2, 3, 4], size = num_agents))

Qin = jnp.asarray(np.random.rand(1, num_agents,4))

Qout, mask = agent.Qoutcomp(Qin, choices)
print(Qin)
print("---")
print(choices)
print("---")
print(Qout)
print("---")
print(mask)

#%%
"Test compute_probs()"
import numpy as np
import model_jax as mj

from jax import random as jran
import jax
from jax import numpy as jnp

num_agents = 1

Q_init_group = []
num_parameters = 6
k = 4.

# key = jran.PRNGKey(900)
key = jran.PRNGKey(np.random.randint(10000))

"Simulate with random parameters"
"Simulate with random parameters"
parameter = jran.uniform(key=key, minval=0, maxval=1,
                         shape=(1, num_agents, 6))

_, key = jran.split(key)
lr_day1 = parameter[..., 0]*0.01 # so lr_day1 has shape (1, num_agents)
theta_Q_day1 = parameter[..., 1]*6
theta_rep_day1 = parameter[..., 2]*6

lr_day2 = parameter[..., 3]*0.01
theta_Q_day2 = parameter[..., 4]*6
theta_rep_day2 = parameter[..., 5]*6

Q_init = jnp.repeat(jnp.asarray([[[0.2, 0., 0., 0.2]]]), num_agents,
                    axis=1)
Q_init_group.append(Q_init)

agent = mj.Vbm_B(lr_day1=jnp.asarray(lr_day1),
              theta_Q_day1=jnp.asarray(theta_Q_day1),
              theta_rep_day1=jnp.asarray(theta_rep_day1),
              lr_day2=jnp.asarray(lr_day2),
              theta_Q_day2=jnp.asarray(theta_Q_day2),
              theta_rep_day2=jnp.asarray(theta_rep_day2),
            errorrates_stt = jnp.asarray(np.random.rand(1, num_agents))*0.1,
            errorrates_dtt = jnp.asarray(np.random.rand(1, num_agents))*0.2,
              k=k,
              Q_init=jnp.asarray(Q_init))

V = jnp.asarray(np.random.rand(1, num_agents, 4))
trial = jnp.asarray(np.random.choice([-1, 1, 2, 3, 4, 12, 13, 14, 23, 24, 34], size = num_agents))

probs = agent.compute_probs(V, trial)

print("--- V:")
print(V)
print("--- trial:")
print(trial)
print("--- probs:")
print(probs)