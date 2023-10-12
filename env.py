#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:16:01 2023

@author: sascha
"""

import pandas as pd

import ipdb
import jax
from jax import random as jran
from jax import numpy as jnp
from jax import lax
import scipy
import numpy as np


class Env():

    def __init__(self, agent, rewprobs, sequence, blockorder, matfile_dir):
        '''

        Parameters
        ----------
        agent : TYPE
            DESCRIPTION.
        rewprobs : TYPE
            DESCRIPTION.
        sequence : list, optional
            list of length num_agents describing the stimulus sequences for the different agents. The default is 1.
            1/2 : default sequence/ mirror sequence
        matfile_dir : TYPE
            DESCRIPTION.
        Returns
        -------
        None.

        '''
        

        assert(isinstance(sequence, list))
        assert(len(sequence) == agent.num_agents)

        self.sequence = sequence
        self.blockorder = blockorder
        self.agent = agent
        self.rewprobs = jnp.asarray(rewprobs)
        self.matfile_dir = matfile_dir

        self.blocktype = self.define_blocktype(blockorder)

    def load_matfiles(self, matfile_dir, blocknr, blocktype, sequence=1):
        '''
        Parameters
        ----------
        matfile_dir : TYPE
            DESCRIPTION.
        blocknr : int
            Number of block of given type (not same as Blockidx in experiment)
        blocktype : TYPE
            DESCRIPTION.
        sequence : list, optional
            list of length num_agents describing the stimulus sequences for the different agents. The default is 1.
            1/2 : default sequence/ mirror sequence

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        list
            DESCRPITION.
        list
            DESCRIPTION.
        jokertypes : list
            DESCRIPTION.

        '''
        
        
        if sequence == 2:
            prefix = "mirror_"
        else:
            prefix = ""

        if blocktype == 0:
            "sequential"
            mat = scipy.io.loadmat(
                matfile_dir + prefix + 'trainblock' + str(blocknr+1) + '.mat')
        elif blocktype == 1:
            "random"
            mat = scipy.io.loadmat(
                matfile_dir + prefix + 'random' + str(blocknr+1) + '.mat')

        else:
            raise Exception("Problem with los blocktypos.")

        seq = mat['sequence'][0]
        seq_no_jokers = mat['sequence_without_jokers'][0]

        "----- Determine congruent/ incongruent jokers ------"
        # -1 no joker
        # 0 random
        # 1 congruent
        # 2 incongruent

        "---- Map Neutral Jokers to 'No Joker' ---"
        seq_noneutral = [t if t != 14 and t != 23 else 1 for t in seq]

        if blocktype == 0:
            "sequential"
            jokers = [-1 if seq_noneutral[tidx] < 10 else seq_no_jokers[tidx]
                      for tidx in range(len(seq))]
            if sequence == 1:
                jokertypes = [j if j == -1 else 1 if j == 1 else 2 if j ==
                              2 else 2 if j == 3 else 1 for j in jokers]

            elif sequence == 2:
                jokertypes = [j if j == -1 else 2 if j == 1 else 1 if j ==
                              2 else 1 if j == 3 else 2 for j in jokers]

            else:
                raise Exception("Fehla!!")

        elif blocktype == 1:
            "random"
            jokertypes = [-1 if seq_noneutral[tidx] <
                          10 else 0 for tidx in range(len(seq_noneutral))]

        return np.squeeze(seq).tolist(), \
            np.squeeze(seq_no_jokers).tolist(), \
            jokertypes
    
    def define_blocktype(self, blockorder):
        '''
        Parameters
        ----------
        blockorder : list
            DESCRIPTION.

        Returns
        -------
        blocktype : list of arrays
            DESCRIPTION.

        '''
        
        assert(self.agent.num_agents == len(blockorder))
        
        blocktype = []
        
        for ag in range(self.agent.num_agents):
            
            num_blocks = 14
            tb_idxs = [1, 3, 5, 6, 8, 10, 12]  # tb idxs for block_order == 1
            # random idxs for block_order == 1
            rand_idxs = [0, 2, 4, 7, 9, 11, 13]
    
            blocktype_temp = np.ones((num_blocks, 480))*-1
    
            dumb = np.array([[0, 1], [1, 0]][blockorder[ag] - 1])
    
            # fixed sequence condition
            blocktype_temp[tb_idxs[0:num_blocks//2], :] = dumb[0]
            # random condition
            blocktype_temp[rand_idxs[0:num_blocks//2], :] = dumb[1]
            
            blocktype.append(blocktype_temp.astype(int).copy())
        
        return blocktype

    def prepare_sims(self):
        "Manipulates self.data for simulations"
        sequence = self.sequence # list of ints
        blocktype = self.blocktype # list of arrays
        num_blocks = 14
        tb_block = [0]*self.agent.num_agents
        random_block = [0]*self.agent.num_agents
        self.data = {"trialsequence": [],
                     "jokertypes": [], 
                     "blockidx": [],
                     "blocktype": []}
        
        for block in range(num_blocks):
            "New block!"
            self.data["trialsequence"].append([-1]*self.agent.num_agents)
            self.data["jokertypes"].append([-1]*self.agent.num_agents)
            self.data["blockidx"].append([-1]*self.agent.num_agents)
            self.data["blocktype"].append([-1]*self.agent.num_agents)
            self.data["blocktype"].extend([[blocktype[i][block, 0] for i in range(self.agent.num_agents)] for _ in range(480)])
            self.data["blockidx"].extend([[block]*self.agent.num_agents]*480)
            
            current_idx = len(self.data["trialsequence"])
            self.data["trialsequence"].extend(np.zeros((480, self.agent.num_agents), dtype=int).tolist())
            self.data["jokertypes"].extend(np.zeros((480, self.agent.num_agents), dtype=int).tolist())

            for ag in range(self.agent.num_agents):
                current_blocktype = blocktype[ag][block, 0]
    
                if current_blocktype == 0:
                    "Sequential block"
                    seq_matlab, seq_no_jokers_matlab, jokertypes = self.load_matfiles(
                                                                    self.matfile_dir, 
                                                                    tb_block[ag], 
                                                                    current_blocktype, 
                                                                    sequence = sequence[ag])
                    tb_block[ag] += 1
    
                    # self.data["blocktype"].extend([[0]]*480)
    
                elif current_blocktype == 1:
                    "Random block"
                    seq_matlab, seq_no_jokers_matlab, jokertypes = self.load_matfiles(
                                                                    self.matfile_dir, 
                                                                    random_block[ag], 
                                                                    current_blocktype, 
                                                                    sequence = sequence[ag])
                    random_block[ag] += 1
                    
                    # self.data["blocktype"].extend([[1]]*480)
    
                else:
                    raise Exception("Problem with los blocktypos.")
                
                for i in range(480):
                    # ipdb.set_trace()
                    self.data["trialsequence"][current_idx + i][ag] = seq_matlab[i]
                    self.data["jokertypes"][current_idx + i][ag] = jokertypes[i]
                # self.data["trialsequence"].extend([[s] for s in seq_matlab])
                # self.data["jokertypes"].extend([[j] for j in jokertypes])
                # ipdb.set_trace()
                
    def run(self, key=None):
        """
        This method is used by simulation()
        
        Parameters
        ----------

        block_order : 1 or 2
            Which block order to use (in case of Context =="all")

        """
        self.prepare_sims()

        # blocktype = jnp.array(self.blocktype)
        num_agents = self.agent.num_agents
        # The index of the block in the current experiment

        def one_trial(carry, matrices):
            "Simulates choices in dual-target trials"
            "matrices is [days, trials.T, lin_blocktype.T]"
            day, trial, blocktype = matrices
            # print("Printing shapes")
            # print(day.shape)
            # print(trial.shape)
            # print(blocktype.shape)
            key, Q, pppchoice, ppchoice, pchoice, seq_counter, rep, V = carry
            # print("bliblibli")
            # print(type(trial))
            # print(trial.shape)
            current_choice, key = self.agent.choose_action(V, trial, key)
            outcome = jran.bernoulli(key, self.rewprobs[current_choice % 4])
            current_choice = current_choice[0, ...]
            outcome = outcome[0, ...]
            _, key = jran.split(key)
            Q, pppchoice, ppchoice, pchoice, seq_counter, rep, V = \
                self.agent.update(jnp.asarray(current_choice),
                              jnp.asarray(outcome), 
                              blocktype,
                              day=day,
                              trial=trial,
                              Q = Q,
                              pppchoice = pppchoice, 
                              ppchoice = ppchoice, 
                              pchoice = pchoice,
                              seq_counter = seq_counter,
                              rep =rep,
                              V = V)
            
            outtie = [current_choice, outcome, Q]
            carry = [key, Q, pppchoice, ppchoice, pchoice, seq_counter, rep, V]
            return carry, outtie
        
        days = (np.array(self.data['blockidx']) > 5) + 1
        trials = jnp.asarray(self.data["trialsequence"])
        blocktype = jnp.asarray(self.data["blocktype"])
        
        # lin_blocktype = jnp.hstack([-jnp.ones((14, 1), dtype=int), blocktype.astype(int)])
        # lin_blocktype = jnp.repeat(lin_blocktype.reshape(-1)[None, ...], num_agents, axis=0)
        # trials = jnp.repeat(trials[None, ...], num_agents, axis=0)
        matrices = [days, trials, blocktype]
        carry = [key, 
                 self.agent.Q, 
                 self.agent.pppchoice,
                 self.agent.ppchoice,
                 self.agent.pchoice,
                 self.agent.seq_counter,
                 self.agent.rep,
                 self.agent.V]
        carry, outties = lax.scan(one_trial, carry, matrices)
        choices, outcomes, Qs = outties
        self.data["choices"] = choices
        self.data["outcomes"] = outcomes
        
        "Add binary choice data (0/1: first/second response option chosen, -1: no dual-target trial)"
        # print("Test this for multiple agents")
        # dfgh
        # option0 = self.agent.find_resp_options(self.data["trialsequence"])[0]
        option1 = self.agent.find_resp_options(self.data["trialsequence"])[1]
        
        "bin_choices: 0/1: 1st/2nd response option"
        "Fill with -1"
        self.data["bin_choices"] = (-1* (self.data['choices'] > -20)).astype(int)
        "Fill with 1s and 0s where option0 or option1 was chosen"
        self.data["bin_choices"] = ((option1 == self.data['choices']) * (self.data['choices'] > -1)).astype(int)
                
        "Create 'bin_choices_w_errors"
        "For single-target trials: 0/1: error/ no error"
        "For dual-target trials: 0: error, 1: 1st response option, 2: 2nd response option"
        self.data["bin_choices_w_errors"] = jnp.where(self.data['choices'] == -2, 
                                                      ~(self.data['choices']==-2), 
                                                      self.data["bin_choices"]+1)   
        
        self.data["bin_choices_w_errors"] = jnp.where(jnp.asarray(self.data['trialsequence']) > 10,
                                                      self.data["bin_choices_w_errors"],
                                                      ~(self.data['choices'] == -2))
        
        self.agent.data = self.data
        self.agent.jitted_one_session = jax.jit(self.agent.one_session)
        return carry, choices, outcomes, Qs

    def envdata_to_df(self):
        
        import ipdb
        import time
        start = time.time()
        num_agents = self.agent.num_agents
        num_trials = len(self.data["trialsequence"])
        df_data = {
        'participantidx': [ag for ag in range(num_agents) for _ in range(num_trials)],
        
        'trialsequence': np.broadcast_to(np.array(self.data['trialsequence']),
                                         (num_trials, num_agents)).reshape(num_trials*num_agents, 
                                                                           order = 'F'),
        
        'jokertypes': np.broadcast_to(np.array(self.data['jokertypes']),
                                         (num_trials, num_agents)).reshape(num_trials*num_agents,
                                                                           order = 'F'),
        
        'blockidx': np.broadcast_to(np.array(self.data['blockidx']),
                                    (num_trials, num_agents)).reshape(num_trials*num_agents,
                                                                      order = 'F'),
        
        'choices': self.data['choices'].reshape(num_trials*num_agents, 
                                                order = 'F'),
        
        'outcomes': self.data['outcomes'].reshape(num_trials*num_agents, 
                                                  order = 'F'),
        }

        sim_df = pd.DataFrame(df_data, dtype=int)
        sim_df['outcomes']
        sim_df = sim_df[sim_df.trialsequence != -1] 
        sim_df.reset_index(drop = True, inplace = True)
        
        "Add column 'trialidx'"
        trialidx = [tt for _ in range(num_agents) for tt in range(len(sim_df)//num_agents)] # as seen in experiment, thus per participant
        sim_df['trialidx'] = trialidx
        
        "Add column 'GDchoice', but don't forget possible errors (-2)!"
        sim_df['GDchoice'] = sim_df.apply(lambda row: 
                                          self.rewprobs[row['choices']] == \
                                          np.max(self.rewprobs) if row['choices'] != -2 else -2, axis = 1).astype(int)
        
            
        end = time.time()
        print("Executed envdata_to_df() in %.2f seconds"%(end-start))
        
        return sim_df
    
    
#%%