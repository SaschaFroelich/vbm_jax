import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import numpyro
numpyro.set_host_device_count(4)

import utils
import ipdb
import numpyro
from numpyro.infer import MCMC, NUTS, SA
import numpyro.distributions as dist
from jax import numpy as jnp
from jax import random, lax
import numpy as np

import model_jax as mj

def define_model(exp_data, 
                 num_agents, 
                 non_dtt_row_indices,
                 errorrates_stt,
                 errorrates_dtt):
    
    """Define priors and likelihood."""
    # num_agents = len(exp_data['choices'][0])
    
    observed = jnp.delete(exp_data['bin_choices_w_errors'], non_dtt_row_indices, axis = 0)
    
    with numpyro.plate('subject', num_agents) as ind:
        
        "First dimension: day, second dimension: agent"
        lrs_day1 = numpyro.sample('lrs_day1', dist.Beta(2, 3))
        lrs_day2 = numpyro.sample('lrs_day2', dist.Beta(2, 3))
        
        theta_q_day1 = numpyro.sample('theta_q_day1', dist.HalfNormal(8))#.expand([2]))
        theta_q_day2 = numpyro.sample('theta_q_day2', dist.HalfNormal(8))#.expand([2]))

        theta_r_day1 = numpyro.sample('theta_r_day1', dist.HalfNormal(8))#.expand([2]))
        theta_r_day2 = numpyro.sample('theta_r_day2', dist.HalfNormal(8))#.expand([2]))
        
        agent = mj.Vbm_B(lr_day1=lrs_day1[None, :], 
                         lr_day2=lrs_day2[None, :], 
                         theta_Q_day1=theta_q_day1[None, :],
                         theta_Q_day2=theta_q_day1[None, :], 
                         theta_rep_day1=theta_r_day1[None, :],
                         theta_rep_day2=theta_r_day2[None, :], 
                         k=4.0,
                         errorrates_stt = errorrates_stt,
                         errorrates_dtt = errorrates_dtt,
                         Q_init=jnp.array([[[0.2, 0, 0, 0.2]]*num_agents]))
        
        # agent = mj.Vbm_B(lr_day1=lrs[0, :][None, :], 
        #                  lr_day2=lrs[1, :][None, :], 
        #                  theta_Q_day1=theta_q[0, :][None, :],
        #                  theta_Q_day2=theta_q[1, :][None, :], 
        #                  theta_rep_day1=theta_r[0, :][None, :],
        #                  theta_rep_day2=theta_r[1, :][None, :], 
        #                  k=4.0,
        #                  errorrates_stt = errorrates_stt,
        #                  errorrates_dtt = errorrates_dtt,
        #                  Q_init=jnp.array([[[0.2, 0, 0, 0.2]]*num_agents]))
        
        # print("What about new block trials?")
        probs = jnp.squeeze(one_session(agent, exp_data))
        "Remove new block trials"
        probabils = jnp.delete(probs, non_dtt_row_indices, axis = 0)
    
        # numpyro.sample('like',
        #                 dist.Categorical(probs=probs), 
        #                 obs=jnp.squeeze(exp_data['bin_choices_w_errors']))
        numpyro.sample('like',
                        dist.Categorical(probs=probabils), 
                        obs=observed)

def one_session(agent, exp_data):
    """Run one entire session with all trials using the jax agent."""
    # self.prepare_sims()

    # num_agents = num_parts
    # The index of the block in the current experiment

    def one_trial(carry, matrices):
        day, trial, blocktype, current_choice, outcome = matrices
        Q, pppchoice, ppchoice, pchoice, seq_counter, rep, V = carry
        
        probs = agent.compute_probs(V, trial)
        
        Q, pppchoice, ppchoice, pchoice, seq_counter, rep, V = \
            agent.update(current_choice,
                          outcome, 
                          blocktype,
                          day = day,
                          trial = trial,
                          Q = Q,
                          pppchoice = pppchoice, 
                          ppchoice = ppchoice, 
                          pchoice = pchoice,
                          seq_counter = seq_counter,
                          rep = rep,
                          V = V)
        
        outtie = [probs]
        carry = [Q, pppchoice, ppchoice, pchoice, seq_counter, rep, V]
        return carry, outtie
    
    days = (np.array(exp_data['blockidx']) > 5) + 1
    trials = np.array(exp_data["trialsequence"])
    blocktype = np.array(exp_data["blocktype"])
    choices = np.array(exp_data['choices'])
    outcomes = np.array(exp_data['outcomes'])
    # lin_blocktype = jnp.hstack([-jnp.ones((14, 1), dtype=int),
    #                             blocktype.astype(int)])
    # lin_blocktype = jnp.repeat(lin_blocktype.reshape(-1)[None, ...],
    #                            num_agents, axis=0)
    # trials = jnp.repeat(trials[None, ...], num_agents, axis=0)
    matrices = [days, trials, blocktype, choices, outcomes]
    carry = [agent.Q, 
             agent.pppchoice,
             agent.ppchoice,
             agent.pchoice,
             agent.seq_counter,
             agent.rep,
             agent.V]
    
    key, probs = lax.scan(one_trial, carry, matrices)

    "Necessarily, probs still contains the values for the new block trials and single-target trials."

    return probs[0]

def perform_inference(exp_data):
    
    num_agents = len(exp_data['trialsequence'][0])
    num_chains = 1
    num_samples = 5
    
    new_block_trials = jnp.nonzero(jnp.squeeze(jnp.asarray(exp_data['trialsequence'])[:,0] == -1))[0]
    choices_wo_newblocktrials = jnp.delete(jnp.asarray(exp_data['choices']), new_block_trials, axis = 0)
    trialseq_wo_newblocktrials = jnp.delete(jnp.asarray(exp_data['trialsequence']), new_block_trials, axis = 0)
    
    num_stt = 5712
    num_dtt = 1008
    
    "Compute error rates"
    ER_stt = jnp.count_nonzero(((trialseq_wo_newblocktrials < 10) * choices_wo_newblocktrials) == -2, axis = 0) / num_stt
    ER_dtt = jnp.count_nonzero(((trialseq_wo_newblocktrials > 10) * choices_wo_newblocktrials) == -2, axis = 0) / num_dtt
    
    "Find rows where no participant saw a dtt"
    non_dtt_row_indices = np.where(np.all(np.array(exp_data['trialsequence']) < 10, axis=1))[0]
    
    kernel = NUTS(define_model, dense_mass=True)
    mcmc = MCMC(kernel, 
                num_warmup=5, 
                num_samples=num_samples,
                num_chains=num_chains, 
                progress_bar=True)
    
    rng_key = random.PRNGKey(1)
    mcmc.run(rng_key, 
             exp_data=exp_data, 
             num_agents=num_agents,
             non_dtt_row_indices = non_dtt_row_indices,
             errorrates_stt = jnp.asarray([ER_stt]),
             errorrates_dtt = jnp.asarray([ER_dtt]))
    mcmc.print_summary()
    
    return mcmc

#%%
# group_data = utils.get_group_data()


