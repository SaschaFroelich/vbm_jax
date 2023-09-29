#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:16:01 2023

@author: sascha
"""

from jax import random as jran
from jax import numpy as jnp
from jax import lax
import scipy
import numpy as np


class env():

    def __init__(self, agent, rewprobs, matfile_dir, sequence=1):
        """sequence : 1 or 2
            2 means mirror sequence
        """
        self.sequence = sequence
        self.agent = agent
        self.rewprobs = jnp.asarray(rewprobs)
        self.matfile_dir = matfile_dir

        self.choices = []
        self.outcomes = []
        self.blocktype = self.define_blocktype()

    def load_matfiles(self, matfile_dir, blocknr, blocktype, sequence=1):
        "blocknr : num. of block of given type (not same as Blockidx in experiment)"
        "sequence : 1 or 2 (2 means mirror sequence)"

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
    
    def define_blocktype(self, block_order=1):
        num_blocks = 14
        tb_idxs = [1, 3, 5, 6, 8, 10, 12]  # tb idxs for block_order == 1
        # random idxs for block_order == 1
        rand_idxs = [0, 2, 4, 7, 9, 11, 13]

        blocktype = np.ones((num_blocks, 480))*-1

        dumb = np.array([[0, 1], [1, 0]][block_order - 1])

        # fixed sequence condition
        blocktype[tb_idxs[0:num_blocks//2], :] = dumb[0]
        # random condition
        blocktype[rand_idxs[0:num_blocks//2], :] = dumb[1]

        return blocktype

    def prepare_sims(self, ):
        "Manipulates self.data for simulations"
        sequence = self.sequence
        blocktype = self.blocktype
        num_blocks = 14
        tb_block = 0
        random_block = 0
        self.data = {"trialsequence": [],
                     "jokertypes": [], "blockidx": []}
        for block in range(num_blocks):
            "New block!"
            self.data["trialsequence"].append([-1])
            self.data["jokertypes"].append([-1])
            self.data["blockidx"].append([block])
            # The index of the block in the current experiment

            current_blocktype = blocktype[block, 0]

            if current_blocktype == 0:
                "Sequential block"
                seq_matlab, seq_no_jokers_matlab, jokertypes = self.load_matfiles(
                    self.matfile_dir, tb_block, current_blocktype, sequence=sequence)
                tb_block += 1

            elif current_blocktype == 1:
                "Random block"
                seq_matlab, seq_no_jokers_matlab, jokertypes = self.load_matfiles(
                    self.matfile_dir, random_block, current_blocktype, sequence=sequence)
                random_block += 1

            else:
                raise Exception("Problem with los blocktypos.")

            self.data["trialsequence"].extend([[s] for s in seq_matlab])
            self.data["jokertypes"].extend([[j] for j in jokertypes])
            self.data["blockidx"].extend([[block]]*480)

    def run(self, key=None):
        """
        This method is used by simulation()
        
        Parameters
        ----------

        block_order : 1 or 2
            Which block order to use (in case of Context =="all")

        """
        self.prepare_sims()

        blocktype = jnp.array(self.blocktype)
        num_agents = self.agent.num_agents
        # The index of the block in the current experiment

        def one_trial(key, matrices):
            "Simulates choices in dual-target trials"
            "matrices is [days, trials.T, lin_blocktype.T]"
            day, trial, blocktype = matrices
            current_choice, key = self.agent.choose_action(trial, day, key)
            outcome = jran.bernoulli(key, self.rewprobs[current_choice % 4])
            current_choice = current_choice[0, ...]
            outcome = outcome[0, ...]
            _, key = jran.split(key)
            self.agent.update(jnp.asarray(current_choice),
                              jnp.asarray(outcome), blocktype,
                              day=day, trial=trial)
            outtie = [current_choice, outcome]
            return key, outtie
        
        days = (np.array(self.data['blockidx']) > 5) + 1
        trials = np.squeeze(self.data["trialsequence"])
        lin_blocktype = jnp.hstack([-jnp.ones((14, 1), dtype=int),
                                    blocktype.astype(int)])
        lin_blocktype = jnp.repeat(lin_blocktype.reshape(-1)[None, ...],
                                   num_agents, axis=0)
        trials = jnp.repeat(trials[None, ...], num_agents, axis=0)
        matrices = [days, trials.T, lin_blocktype.T]
        key, outties = lax.scan(one_trial, key, matrices)
        choices, outcomes = outties
        self.choices = choices
        self.outcomes = outcomes
        
        return key

    def one_session(self, choices, outcomes, trials, blocktype, num_parts, key=None):
        """
        Should simulate choices if no choices are given, and compute probs if choices are given.
        
        Parameters
        ----------

        block_order : 1 or 2
            Which block order to use (in case of Context =="all")

        """
        self.prepare_sims()

        num_agents = num_parts
        # The index of the block in the current experiment

        def one_trial(key, matrices):
            "Computes response probabilities for dual-target trials"
            "matrices is [days, trials.T, lin_blocktype.T, choices, outcomes]"
            day, trial, blocktype, current_choice, outcome = matrices
            probs = self.agent.compute_probs(trial, day)
            _, key = jran.split(key)
            self.agent.update(jnp.asarray(current_choice),
                              jnp.asarray(outcome), blocktype,
                              day=day, trial=trial)
            outtie = [probs]
            return key, outtie
        
        days = (np.array(self.data['blockidx']) > 5) + 1
        trials = np.squeeze(self.data["trialsequence"])
        lin_blocktype = jnp.hstack([-jnp.ones((14, 1), dtype=int),
                                    blocktype.astype(int)])
        lin_blocktype = jnp.repeat(lin_blocktype.reshape(-1)[None, ...],
                                   num_agents, axis=0)
        trials = jnp.repeat(trials[None, ...], num_agents, axis=0)
        matrices = [days, trials.T, lin_blocktype.T, choices, outcomes]
        key, probs = lax.scan(one_trial, key, matrices)
        return probs
