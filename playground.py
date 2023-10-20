#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:13:46 2023

@author: sascha
"""

"Simulate behaviour"
import model_jax as mj
from jax import numpy as jnp
import numpy as np
import jax

num_agents = 2

theta_rep_day1 = 200
theta_rep_day2 = 0
theta_Q_day1 = 3.
theta_Q_day2 = 3.
# lr_day1 = 0.005
lr_day1 = 0.00
lr_day2 = 0.

if 0:
    locs = jnp.array([jax.scipy.special.logit(lr_day1/0.05),
                    jnp.log(theta_Q_day1),
                    jnp.log(theta_rep_day1),
                    jax.scipy.special.logit(lr_day2/0.05),
                    jnp.log(theta_Q_day2),
                    jnp.log(theta_rep_day2)])
    
    locs = locs.repeat(num_agents).reshape(num_agents,6, order ='F')
    
    sim_df, env_data, agent = mj.simulation(num_agents = num_agents, 
                                      sequence = [1]*num_agents,
                                      blockorder = [1]*num_agents,
                                      locs = locs[None, ...],
                                      DataFrame = True)
else:
    _, env_data, agent = mj.simulation(num_agents = num_agents, 
                                      sequence = [1]*num_agents,
                                      blockorder = [1]*num_agents,

                                errorrates_stt = jnp.ones((1,num_agents))*0.1,
                                errorrates_dtt = jnp.ones((1,num_agents))*0.2,
                                lr_day1 = jnp.asarray([[lr_day1]*num_agents]),
                                lr_day2 = jnp.asarray([[lr_day2]*num_agents]), 
                                
                                theta_Q_day1 = jnp.asarray([[theta_Q_day1]*num_agents]),
                                theta_Q_day2 = jnp.asarray([[theta_Q_day2]*num_agents]),
                                
                                theta_rep_day1 = jnp.asarray([[theta_rep_day1]*num_agents]),
                                theta_rep_day2 = jnp.asarray([[theta_rep_day2]*num_agents]),
                                DataFrame = False)



# choices = outties[0]
# outcomes = outties[1]
# Qs = outties[2][0]

'''
Jokertypes:
    -1 no joker
    0 random
    1 congruent
    2 incongruent
'''
mj.plot_simulated(sim_df)


sim_df_temp = sim_df[sim_df.jokertypes != -1]
print(np.unique(sim_df_temp['choices']))

#%%
from jax import random as jran

Q_init_group = []
npar = 6
k = 4.

key = jran.PRNGKey(np.random.randint(10000))

"Simulate with random parameters"
parameter = jran.uniform(key=key, minval=0, maxval=1,
                         shape=(1, num_agents, npar))
_, key = jran.split(key)
lr_day1 = parameter[..., 0]*0.01 # so lr_day1 has shape (1, num_agents)
theta_Q_day1 = parameter[..., 1]*6
theta_rep_day1 = parameter[..., 2]*6

lr_day2 = parameter[..., 3]*0.01
theta_Q_day2 = parameter[..., 4]*6
theta_rep_day2 = parameter[..., 5]*6

errorrates_stt = parameter[..., 7]
errorrates_dtt = parameter[..., 6]


Q_init = jnp.repeat(jnp.asarray([[[0.2, 0., 0., 0.2]]]), 
                    num_agents,
                    axis=1)

Q_init_group.append(Q_init)

agent = mj.Vbm_B(lr_day1=jnp.asarray(lr_day1),
              theta_Q_day1=jnp.asarray(theta_Q_day1),
              theta_rep_day1=jnp.asarray(theta_rep_day1),
              lr_day2=jnp.asarray(lr_day2),
              theta_Q_day2=jnp.asarray(theta_Q_day2),
              theta_rep_day2=jnp.asarray(theta_rep_day2),
              k=k,
              errorrates_stt = jnp.asarray(errorrates_stt),
              errorrates_dtt = jnp.asarray(errorrates_dtt),
              Q_init=jnp.asarray(Q_init))

trial = jnp.asarray([2,12,34,4])

probs = agent.compute_probs(agent.V, trial)

current_choice, _ = agent.choose_action(agent.V, trial, key)

print(current_choice)

#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

for ag in range(num_agents):
    sim_df_temp = sim_df[sim_df['participantidx'] == ag]
    sim_df_temp = sim_df_temp[sim_df_temp.jokertypes != -1]
    grouped = sim_df_temp.groupby(['blockidx', 'jokertypes'])
    average = grouped['GDchoice'].mean()
    
    new_df = pd.DataFrame(average)
    fig,ax = plt.subplots()
    ax1 = sns.lineplot(data=new_df, x="blockidx", y="GDchoice", hue="jokertypes")
    plt.ylim([0,1 ])
    # lines = ax1.get_lines()
    # ipdb.set_trace()
    # min_y = np.min([line.get_ydata().min() for line in lines])
    # plt.plot([5.5, 5.5], [0.6, 1], color = 'k')
    plt.show()



#%%
import ipdb

def repeat_interleave(x, num):
    return jnp.hstack([x[:, None]] * num).reshape(-1)

def Qoutcomp(Qin, choices):
    '''
        Parameters:
            Qin : torch.tensor() with shape [num_particles, num_agents, 4]
            choices :   torch.tensor() with shape [num_agents]

        Returns:
            Returns a tensor Qout with the same shape as Qin. 
            Qout contains zeros everywhere except for the positions indicated 
            by the choices of the participants. In this case the value of Qout is that of Qin.
            Qout contains 0s for choices indicated by -10 (error choice for example), 
            and will thus be ignored in Q-value updating.
    '''
    bad_choice = -2

    if len(Qin.shape) == 2:
        Qin = Qin[None, ...]

    elif len(Qin.shape) == 3:
        pass

    else:
        ipdb.set_trace()
        raise Exception("Fehla, digga!")

    # no_error_mask = [1 if ch != -10 else 0 for ch in choices]
    no_error_mask = jnp.array(choices) != bad_choice
    "Replace error choices by the number one"
    choices_noerrors = jnp.where(jnp.asarray(no_error_mask, dtype=bool),
                                 choices, jnp.ones(choices.shape)).astype(int)

    Qout = jnp.zeros(Qin.shape, dtype=float)
    choicemask = jnp.zeros(Qin.shape, dtype=int)
    num_particles = Qout.shape[0]  # num of particles
    num_agents = Qout.shape[1]  # num_agents

    # errormask = jnp.asarray([0 if c == -10 else 1 for c in choices])
    errormask = jnp.array(choices) != bad_choice
    errormask = jnp.broadcast_to(
        errormask, (num_particles, 4, num_agents)).transpose(0, 2, 1)

    x = jnp.arange(num_particles).repeat(num_agents)
    y = repeat_interleave(jnp.arange(num_agents), num_particles)
    z = repeat_interleave(choices_noerrors, num_particles)
    Qout = Qout.at[x, y, z].set(Qin[x, y, z])  # TODO: Improve speed

    # choicemask[repeat_interleave(jnp.arange(num_particles), num_agents),
    #            jnp.arange(num_agents).repeat(num_particles),
    #            choices_noerrors.repeat(num_particles)] = 1
    choicemask = choicemask.at[repeat_interleave(jnp.arange(num_particles), num_agents),
                               jnp.arange(num_agents).repeat(
                                   num_particles),
                               choices_noerrors.repeat(num_particles)].set(1)

    mask = errormask*choicemask
    return Qout*mask, mask


#%%

Qin = jnp.asarray([[[0.2, 0., 0., 0.2],[0.2, 0., 0., 0.2]]])
choices = jnp.asarray([-2, 3])
Qout, mask = Qoutcomp(Qin, choices)

#%%
import numpy as np
from jax import random as jran

key = jran.PRNGKey(np.random.randint(10000))
    
npar = 6
k = 4.0

lr_day1_true = []
theta_Q_day1_true = []
theta_rep_day1_true = []

lr_day2_true = []
theta_Q_day2_true = []
theta_rep_day2_true = []
Q_init_group = []
# groupdata = []

"Simulate with random parameters"
parameter = jran.uniform(key=key, minval=0, maxval=1,
                         shape=(1, num_agents, npar))
_, key = jran.split(key)
lr_day1 = parameter[..., 0]*0.01 # so lr_day1 has shape (1, num_agents)
theta_Q_day1 = parameter[..., 1]*6
theta_rep_day1 = parameter[..., 2]*6

lr_day2 = parameter[..., 3]*0.01
theta_Q_day2 = parameter[..., 4]*6
theta_rep_day2 = parameter[..., 5]*6

lr_day1_true.append(lr_day1)
theta_Q_day1_true.append(theta_Q_day1)
theta_rep_day1_true.append(theta_rep_day1)

lr_day2_true.append(lr_day2)
theta_Q_day2_true.append(theta_Q_day2)
theta_rep_day2_true.append(theta_rep_day2)

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

#%%

def repeat_interleave(x, num):
    '''
    Parameters
    ----------
    x : jnp array
        range from 1 to num_agents
    num : int
        Number of particles


    '''
    
    return jnp.hstack([x[:, None]] * num).reshape(-1)

#%%

from jax import random as jran
key = jran.PRNGKey(np.random.randint(10000))

sampled = jran.uniform(key, shape = (4, ))

sampled > errorrates_stt

#%%
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

#%%
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
# import model_jax as mj
import model_jax_old as mj
print("Simulating data for %d agents."%num_agents)
_, exp_data_old, agent = mj.simulation(num_agents = num_agents,
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
np.all(exp_data_new['choices']==exp_data_old['choices'])
np.where(exp_data_new['choices']!=exp_data_old['choices'])
         
#%%
for key in exp_data.keys():
    print(key)
    assert(np.all(exp_data_new[key]==exp_data_old[key]))
    
len(np.where(exp_data_new['choices']!=exp_data_old['choices'])[0])

#%%

"""
Comparison within same file
Compare outputs of new vmapped update() with old update
"""

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
seq_counter =  4. / 4 * jnp.ones((num_agents, 2, 6, 6, 6, 6))
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


#%%
import numpy as np
import model_jax as mj

from jax import random as jran
import jax
from jax import numpy as jnp

num_agents = 2

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

# choices = np.array([0,1,-2,1,0,2,3,2,3,-2])
choices = np.array([1, 0])

Qin = np.random.rand(1, num_agents,4)

Qout, mask = agent.Qoutcomp(Qin, choices)