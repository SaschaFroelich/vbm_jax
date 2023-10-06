# import ipdb
from itertools import product

import ipdb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from jax import numpy as jnp
from jax import lax, nn
from jax import random as jran
import jax

import env

class Vbm():
    "parameters: omega, dectemp, lr"

    def __init__(self, k, Q_init, **kwargs):
        """
        --- Parameters ---
        **kwargs: omega, dectemp, lr
        omega (between 0 & 1): weighting factor between habitual and goal-directed: p(a1) = σ(β*[(1-ω)*(r(a1)-r(a2)) + ω*(Q(a1)-Q(a2)))]
        dectemp (between 0 & inf): decision temperature β
        lr (between 0 & 1) : learning rate
        Q_init : (torch tensor shape [num_particles, num_agents, 4]) initial Q-Values"""

        "---- General Setup (the same for every model)----"
        assert (Q_init.ndim == 3)
        self.num_blocks = 14
        self.trials = 480*self.num_blocks
        self.na = 4  # no. of possible actions

        self.num_particles = Q_init.shape[0]
        self.num_agents = Q_init.shape[1]
        self.pppchoice = -1 * jnp.ones(self.num_agents, dtype=int)
        self.ppchoice = -1 * jnp.ones(self.num_agents, dtype=int)
        self.pchoice = -1 * jnp.ones(self.num_agents, dtype=int)

        "Q and rep"
        self.Q_init = Q_init
        self.Q = [Q_init]  # Goal-Directed Q-Values
        # habitual values (repetition values)
        self.rep = [
            jnp.ones((self.num_particles, self.num_agents, self.na))/self.na]

        "K"
        self.k = k
        self.bad_choice = -2

        # self.posterior_actions = [] # 2 entries: [p(option1), p(option2)]
        # Compute prior over sequences of length 4
        self.init_seq_counter = {}
        "-1 in seq_counter for beginning of blocks (so previos sequence is [-1,-1,-1])"
        "-2 in seq_counter for errors"
        self.init_seq_counter = self.k / 4 * jnp.ones((2, 6, 6, 6, 6,
                                                       self.num_agents))

        self.seq_counter = self.init_seq_counter.copy()
        for key in kwargs:
            assert (kwargs[key].ndim == 2)

        "Latent variables"
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.specific_init(**kwargs)

    def specific_init(self, **kwargs):
        "Model-specific init function (due to input to update_V())"
        "Compute action values V"
        self.param_names = ["omega", "dectemp", "lr"]

        for par in self.param_names:
            assert (par in kwargs)

        self.V = []
        self.update_V()

    def update_V(self):
        "Model-specific function"
        "V(ai) = (1-ω)*rep_val(ai) + ω*Q(ai)"
        # V-Values for actions (i.e. weighted action values)
        V0 = (1-self.omega)*self.rep[-1][..., 0] + \
            self.omega*self.Q[-1][..., 0]
        V1 = (1-self.omega)*self.rep[-1][..., 1] + \
            self.omega*self.Q[-1][..., 1]
        V2 = (1-self.omega)*self.rep[-1][..., 2] + \
            self.omega*self.Q[-1][..., 2]
        V3 = (1-self.omega)*self.rep[-1][..., 3] + \
            self.omega*self.Q[-1][..., 3]
            
        self.V.append(jnp.stack((V0, V1, V2, V3), 2))

    def locs_to_pars(self, locs):
        par_dict = {"omega": nn.sigmoid(locs[..., 0]),
                    "dectemp": jnp.exp(locs[..., 1]),
                    "lr": 0.05*nn.sigmoid(locs[..., 2])}

        for key in par_dict:
            assert (key in self.param_names)

        return par_dict

    def compute_probs(self, V, trial):
        '''
        
        Parameters
        ----------
        V : list containing array, shape [num_particles, num_agents, 4]
            
        trial : array with shape [num_agents]
            DESCRIPTION.
            
        day : array with shape [1]
            DESCRIPTION.

        Returns
        -------
        probs : TYPE
            DESCRIPTION.

        '''
        
        option1, option2 = self.find_resp_options(trial)

        "Replace both response options with 1 for those participants who did not see a joker trial"
        cond1 = jnp.array(option1) > -1
        cond2 = jnp.array(option2) > -1

        option1 = 1 + (-1 + option1) * cond1
        option2 = 1 + (-1 + option2) * cond2

        # option1 = jnp.asarray(
        #     [option1[ii] if option1[ii] > -1 else 1 for ii in range(len(option1))])
        # option2 = jnp.asarray(
        #     [option2[ii] if option2[ii] > -1 else 1 for ii in range(len(option2))])

        _, mask1 = self.Qoutcomp(V[-1], option1)
        # Vopt1 = (self.V[-1][jnp.where(mask1 == 1)]  # (particles, participants, 4)
        #          ).reshape(self.num_particles, self.num_agents)
        _, mask2 = self.Qoutcomp(V[-1], option2)
        # Vopt2 = self.V[-1][jnp.where(mask2 == 1)
        #                    ].reshape(self.num_particles, self.num_agents)
        ind1_last = jnp.argsort(mask1, axis=-1)[:, :, -1].reshape(-1)
        ind_all = jnp.array(list(product(jnp.arange(self.num_particles),
                                         jnp.arange(self.num_agents))))
        ind2_last = jnp.argsort(mask2, axis=-1)[:, :, -1].reshape(-1)
        Vopt1 = V[-1][ind_all[:, 0], ind_all[:, 1], ind1_last]
        Vopt2 = V[-1][ind_all[:, 0], ind_all[:, 1], ind2_last]
        probs = self.softmax(jnp.stack((Vopt1, Vopt2), -1))
        
        "Manipulate 'probs' here to contain the adjusted probabilities"
        probs_prime = jnp.where(np.ones(probs.shape) * trial[None,:][...,None] > 10,
                          probs * (1-self.errorrates_dtt)[..., None],
                          probs * (1-self.errorrates_stt)[..., None])

        "Ers: array with error rates for stt/ dtt"
        ers = jnp.where(trial > 10, self.errorrates_dtt, self.errorrates_stt)[..., None]*np.ones((1, self.num_agents, 1))
        
        "Concatenate ers with the adjusted probs"
        probs_new = jnp.concatenate((ers, probs), axis=2)
        
        # probs = jnp.concatenate(  , probs)

        return probs_new

    def Qoutcomp(self, Qin, choices):
        '''
        
            Parameters:
                Qin : torch.tensor() with shape [num_particles, num_agents, 4]
                choices :   torch.tensor() with shape [num_agents]

            Returns:
                Returns a tensor Qout with the same shape as Qin. 
                Qout contains zeros everywhere except for the positions indicated 
                by the choices of the participants. In this case the value of Qout is that of Qin.
                Qout contains 0s for choices indicated by -2 (error choice for example), 
                and will thus be ignored in Q-value updating.
        '''

        if len(Qin.shape) == 2:
            Qin = Qin[None, ...]

        elif len(Qin.shape) == 3:
            pass

        else:
            ipdb.set_trace()
            raise Exception("Fehla, digga!")

        # no_error_mask = [1 if ch != -10 else 0 for ch in choices]
        no_error_mask = jnp.array(choices) != self.bad_choice
        "Replace error choices by the number one"
        choices_noerrors = jnp.where(jnp.asarray(no_error_mask, dtype=bool),
                                     choices, jnp.ones(choices.shape)).astype(int)

        Qout = jnp.zeros(Qin.shape, dtype=float)
        choicemask = jnp.zeros(Qin.shape, dtype=int)
        num_particles = Qout.shape[0]  # num of particles
        num_agents = Qout.shape[1]  # num_agents

        # errormask = jnp.asarray([0 if c == -10 else 1 for c in choices])
        errormask = jnp.array(choices) != self.bad_choice
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

    def softmax(self, z):
        # sm = nn.softmax(dim=-1)
        # p_actions = sm(self.dectemp[..., None]*z)
        # return p_actions
        return nn.softmax(self.dectemp[..., None] * z, axis=-1)

    def find_resp_options(self, stimulus_mat):
        """
        Given a dual-target stimulus (e.g. 12, 1-indexed), this function returns the two response
        options in 0-indexing. E.g.: stimulus_mat=14 -> option1_python = 0, option1_python = 3
        INPUT: stimulus in MATLAB notation (1-indexed) (simple list of stimuli)
        OUTPUT: response options in python notation (0-indexed)
        """
        stimulus_mat = jnp.array(stimulus_mat, ndmin=1)

        option2_python = (jnp.asarray(stimulus_mat, dtype=int) % 10) - 1
        option1_python = (((jnp.asarray(stimulus_mat) -
                          (jnp.asarray(stimulus_mat) % 10)) / 10) - 1).astype(int)

        if option2_python.ndim == 2 and option2_python.shape[0] == 1:
            option1_python = jnp.squeeze(option1_python)
            option2_python = jnp.squeeze(option2_python)
        cond = jnp.array(stimulus_mat) > 10
        option1_python = self.bad_choice + \
            (-self.bad_choice + option1_python) * cond
        option2_python = self.bad_choice + \
            (-self.bad_choice + option2_python) * cond
        return option1_python, option2_python

    def choose_action(self, V, trial, key):
        "INPUT: trial (in 1-indexing (i.e. MATLAB notation))"
        "OUTPUT: choice response digit (in 0-indexing notation)"
        "Errorrate applies to single-target trials as well as to dual-target trials."
        # assert trial > -1, "Sascha dumb"
        sampled = jran.uniform(key, shape = (self.num_agents, ))
        _, key = jran.split(key)
        cond_error = jnp.where(trial > 10, sampled > self.errorrates_dtt, sampled > self.errorrates_stt) 
        print("cond error")
        print(cond_error.shape)
        cond_trial = trial < 10
        "Dual-target trial"
        option1, option2 = self.find_resp_options(trial)
        probs = self.compute_probs(V, trial)
        choice_sample = jran.uniform(key, shape=probs.shape[:2]) > probs[:, :, 0]
        _, key = jran.split(key)
        choice_python = option2 * choice_sample + \
            option1 * (1-choice_sample)
        choice_if_noerror = (trial - 1) * cond_trial + \
            choice_python * (1 - cond_trial)
        actual_choice = self.bad_choice * \
            (1 - cond_error) + choice_if_noerror * cond_error
            
        return actual_choice, key

class Vbm_B(Vbm):
    "parameters: lr_day1, theta_Q_day1, theta_rep_day1, lr_day2, theta_Q_day2, theta_rep_day2"

    def specific_init(self, **kwargs):
        "Model-specific init function"
        "Compute action values V"

        self.param_names = ["lr_day1", 
                            "theta_Q_day1",
                            "theta_rep_day1", 
                            "lr_day2", 
                            "theta_Q_day2", 
                            "theta_rep_day2"]
        
        for par in self.param_names:
            assert (par in kwargs)

        self.dectemp = jnp.asarray([[1.]])
        self.V = []
        self.update_V(day=1, rep=self.rep, Q=self.Q)

    def update_V(self, day, rep, Q):
        "Model-specific function"
        "V(ai) = Θ_r*rep_val(ai) + Θ_Q*Q(ai)"

        theta_rep = jnp.array([self.theta_rep_day1,
                               self.theta_rep_day2])[1 * (day == 2)]
        
        theta_Q = jnp.array([self.theta_Q_day1, 
                             self.theta_Q_day2])[1 * (day == 2)]
        

        # V-Values for actions (i.e. weighted action values)
        V0 = theta_rep * rep[-1][..., 0] + theta_Q*Q[-1][..., 0]
        V1 = theta_rep * rep[-1][..., 1] + theta_Q*Q[-1][..., 1]
        V2 = theta_rep * rep[-1][..., 2] + theta_Q*Q[-1][..., 2]
        V3 = theta_rep * rep[-1][..., 3] + theta_Q*Q[-1][..., 3]

        V = [jnp.stack((V0, V1, V2, V3), 2)]
        self.V = [jnp.stack((V0, V1, V2, V3), 2)]
        
        return V

    def locs_to_pars(self, locs):
        "Model-specific function"
        par_dict = {"lr_day1": 0.05*nn.sigmoid(locs[..., 0]),
                    "theta_Q_day1": jnp.exp(locs[..., 1]),
                    "theta_rep_day1": jnp.exp(locs[..., 2]),

                    "lr_day2": 0.05*nn.sigmoid(locs[..., 3]),
                    "theta_Q_day2": jnp.exp(locs[..., 4]),
                    "theta_rep_day2": jnp.exp(locs[..., 5])}

        for key in par_dict:
            assert (key in self.param_names)

        return par_dict

    def update(self, 
               choices, 
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
               **kwargs):
        '''
        Is called after a dual-target choice and after a single-target choice.
        Updates Q-values, sequence counters, habit values (i.e. repetition values), and V-Values.

        
        Parameters
        ----------
        choices : array, shape [num_agents]
            The particiapnt's choice at the dual-target trial (0-indexed). -2 & -10 are errors.
            -1 means new block trial
            
        outcome : array, shape [num_agents]
            0/1 : no reward/ reward.
            -1 means new block trial
            
        blocktype : array, shape [num_agents]
            0/1 : sequential/ random 
            
        day : array, shape [1]
            Identical across participants.
            
        trial : array, shape [num_agents]
            The observed trial depends on the experimental group and is not identical across participants.
            
        Q : list, containing array, shape [num_particles, num_agents, 4]
            The Q-values.
            
        pppchoice : array, shape [num_agents]
            DESCRIPTION.
            
        ppchoice : array, shape [num_agents]
            DESCRIPTION.
            
        pchoice : array, shape [num_agents]
            DESCRIPTION.
            
        seq_counter : array, shape [2, 6, 6, 6, 6, num_agents]
            Dimension 0: 0/1 sequential/ random condition
            Dimensions 1-4: (-2, -1, 0, 1, 2, 3), -2 error, -1 new block trial, 0-3 response digits
            Dimension 5: agent
            
        rep : list, containing array, shape [num_particles, num_agents, 4]
            rep[-1] is array with the repetition (i.e. habit) values.
            
        V : list, containing array, shape [num_particles, num_agents, 4]
            V[-1] is array containing the action values for the next trial

        '''
        # print("Printing at beginning of update()")
        # print(choices.shape)
        # print(outcome.shape)
        # print(blocktype.shape)
        # print(day.shape)
        # print(trial.shape)
        # print(Q[-1].shape)
        # print(pppchoice.shape)
        # print(ppchoice.shape)
        # print(pchoice.shape)
        # print("seqcounter")
        # print(seq_counter.shape)
        # print(rep[-1].shape)
        # print(V[-1].shape)
        
        lr = self.lr_day1 * (day == 1) + \
            self.lr_day2 * (day == 2)
            
        pppchoice = (1 - (trial == -1)) * pppchoice - (trial == -1)
        ppchoice = (1 - (trial == -1)) * ppchoice - (trial == -1)
        pchoice = (1 - (trial == -1)) * pchoice - (trial == -1)
        "----- Update GD-values -----"
        # Outcome is either 0 or 1

        "--- Group!!! ----"
        "mask contains 1s where Qoutcomp() contains non-zero entries"
        Qout, mask = self.Qoutcomp(Q[-1], choices)
        Qnew = Q[-1] + lr[..., None] * \
            (outcome[None, ..., None]-Qout)*mask
        
        # Qnew = self.Q[-1]*2
        # Q = [Qnew]
        
        Q = [Qnew * (trial[0] != -1) + Q[-1] * (trial[0] == -1)]
        "--- The following is executed in case of correct and incorrect responses ---"
        "----- Update sequence counters and repetition values of self.rep -----"
        repnew = []
        for agent in range(self.num_agents):
            new_row = [0., 0., 0., 0.]

            "Sequential Block"
            " Update counter "
            old_seq_counter = seq_counter[blocktype[agent],
                                               pppchoice[agent],
                                               ppchoice[agent],
                                               pchoice[agent],
                                               choices[agent], agent]
            seq_counter = seq_counter.at[blocktype[agent],
                                                   pppchoice[agent],
                                                   ppchoice[agent],
                                                   pchoice[agent],
                                                   choices[agent], agent].set(old_seq_counter + 1)

            " Update rep values "
            index = (blocktype[agent], 
                     ppchoice[agent],
                     pchoice[agent], 
                     choices[agent])
            
            for aa in range(4):
                new_row[aa] = seq_counter[index + (aa, agent)] / \
                    (seq_counter[index + (0, agent)] +
                     seq_counter[index + (1, agent)] +
                     seq_counter[index + (2, agent)] +
                     seq_counter[index + (3, agent)])

            repnew.append(new_row)
        jrepnew = jnp.asarray(repnew)
        new_rep = jnp.broadcast_to(jrepnew[None, ...],
                                   (self.num_particles,
                                    self.num_agents, 4))
        rep = [new_rep * (trial[None, :, None] != -1) +
                    jnp.ones((self.num_particles,
                              self.num_agents,
                              self.na))/self.na * (trial[None, :, None] == -1)]

        "----- Compute new V-values for next trial -----"
        V = self.update_V(day=day[0], rep=rep, Q=Q)

        "----- Update action memory -----"
        # pchoice stands for "previous choice"
        pppchoice = ppchoice * (trial != -1) - (trial == -1)
        ppchoice = pchoice * (trial != -1) - (trial == -1)
        pchoice = choices * (trial != -1) - (trial == -1)
        
        return Q, pppchoice, ppchoice, pchoice, seq_counter, rep, V


def simulation(num_agents=100, key=None, **kwargs):
    '''
    Simulates agent behaviour for n agents in the same experiment 
    (i.e. all experimental parameters identical for each agent)

            Parameters:
                    num_agents (int): Number of agents to simulate behaviour for
                    **kwargs : Model parameters (each a jax array with shape num_particles, num_agents)

            Returns:
                    out (DataFrame): Experimental parameters and data
                        'participantidx'
                        'trialsequence'
                        'trialidx'
                        'jokertypes'
                        'blockidx'
                        'choices'
                        'outcomes'
    '''
    
    if key is None:
        key = jran.PRNGKey(np.random.randint(10000))
        
    npar = 8
    k = 4.0

    Q_init_group = []
    # groupdata = []
    
    if 'lr_day1' in kwargs:
        lr_day1 = kwargs['lr_day1'] # so lr_day1 has shape (1, num_agents)
        theta_Q_day1 = kwargs['theta_Q_day1']
        theta_rep_day1 = kwargs['theta_rep_day1']
    
        lr_day2 = kwargs['lr_day2']
        theta_Q_day2 = kwargs['theta_Q_day2']
        theta_rep_day2 = kwargs['theta_rep_day2']
    
        errorrates_stt = kwargs['errorrates_stt']
        errorrates_dtt = kwargs['errorrates_dtt']
        
    else:
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
        
        errorrates_stt = parameter[..., 7]*0.1
        errorrates_dtt = parameter[..., 6]*0.2

    if 'sequence' in kwargs:
        sequence = kwargs['sequence']
    else:
        sequence = np.random.randint(1, 3, size=num_agents).tolist()
        
    if 'blockorder' in kwargs:
        blockorder = kwargs['blockorder']
    else:
        blockorder = np.random.randint(1, 3, size=num_agents).tolist()
        

    Q_init = jnp.repeat(jnp.asarray([[[0.2, 0., 0., 0.2]]]), 
                        num_agents,
                        axis=1)
    
    Q_init_group.append(Q_init)

    agent = Vbm_B(lr_day1=jnp.asarray(lr_day1),
                  theta_Q_day1=jnp.asarray(theta_Q_day1),
                  theta_rep_day1=jnp.asarray(theta_rep_day1),
                  lr_day2=jnp.asarray(lr_day2),
                  theta_Q_day2=jnp.asarray(theta_Q_day2),
                  theta_rep_day2=jnp.asarray(theta_rep_day2),
                  k=k,
                  errorrates_stt = jnp.asarray(errorrates_stt),
                  errorrates_dtt = jnp.asarray(errorrates_dtt),
                  Q_init=jnp.asarray(Q_init))
    
    print("Here, baby!")
    print(agent.V[-1].shape)
    
    newenv = env.Env(agent, 
                     rewprobs=[0.8, 0.2, 0.2, 0.8],
                     sequence = sequence,
                     blockorder = blockorder,
                     matfile_dir='./matlabcode/clipre/')
    
    key, *outties  = newenv.run(key=key)
    
    out_df = newenv.envdata_to_df()
        
    return out_df, newenv.data

def comp_groupdata(groupdata, for_ddm=1):
    """Trialsequence no jokers and RT are only important for DDM to determine the jokertype"""

    if for_ddm:
        newgroupdata = {"Trialsequence": [],\
                        # "Trialsequence no jokers" : [],\
                        "Choices": [],\
                        "Outcomes": [],\
                        "Blocktype": [],\
                        "Blockidx": [],\
                        "RT": []}

    else:
        newgroupdata = {"Trialsequence": [],\
                        # "Trialsequence no jokers" : [],\
                        "Choices": [],\
                        "Outcomes": [],\
                        "Blocktype": [],\
                        "Blockidx": []}

    for trial in range(len(groupdata[0]["Trialsequence"])):
        trialsequence = []
        # trialseq_no_jokers = []
        choices = []
        outcomes = []
        # blocktype = []
        blockidx = []
        if for_ddm:
            RT = []

        for dt in groupdata:
            trialsequence.append(dt["Trialsequence"][trial][0])
            # trialseq_no_jokers.append(dt["Trialsequence no jokers"][trial][0])
            choices.append(dt["Choices"][trial][0])
            outcomes.append(dt["Outcomes"][trial][0])
            # blocktype.append(dt["Blocktype"][trial][0])
            blockidx.append(dt["Blockidx"][trial][0])
            if for_ddm:
                RT.append(dt["RT"][trial][0])

        newgroupdata["Trialsequence"].append(trialsequence)
        # newgroupdata["Trialsequence no jokers"].append(trialseq_no_jokers)
        newgroupdata["Choices"].append(jnp.array(choices, dtype=int))
        newgroupdata["Outcomes"].append(jnp.array(outcomes, dtype=int))
        # newgroupdata["Blocktype"].append(blocktype)
        newgroupdata["Blockidx"].append(blockidx)
        if for_ddm:
            newgroupdata["RT"].append(RT)

    return newgroupdata

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

def plot_simulated(sim_df):
    '''
    Jokertypes:
        -1 no joker
        0 random
        1 congruent
        2 incongruent
    '''
    "First remove entries where jokertypes == -1"
    sim_df = sim_df[sim_df.jokertypes != -1]
    grouped = sim_df.groupby(['blockidx', 'jokertypes'])
    average = grouped['GDchoice'].mean()
    
    new_df = pd.DataFrame(average)
    fig,ax = plt.subplots()
    ax1 = sns.lineplot(data=new_df, x="blockidx", y="GDchoice", hue="jokertypes")
    # lines = ax1.get_lines()
    # ipdb.set_trace()
    # min_y = np.min([line.get_ydata().min() for line in lines])
    # plt.plot([5.5, 5.5], [0.6, 1], color = 'k')
    plt.show()