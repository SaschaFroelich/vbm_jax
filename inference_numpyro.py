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
    
    "First dimension: day, second dimension: agent"
    lrs = numpyro.sample('lrs', dist.Beta(2, 3).expand([2, num_agents]))
    theta_q = numpyro.sample('theta_q', dist.HalfNormal(8).expand([2, num_agents]))
    theta_r = numpyro.sample('theta_r', dist.HalfNormal(8).expand([2, num_agents]))
    
    agent = mj.Vbm_B(lr_day1=lrs[0, :][None, :], 
                     lr_day2=lrs[1, :][None, :], 
                     theta_Q_day1=theta_q[0, :][None, :],
                     theta_Q_day2=theta_q[1, :][None, :], 
                     theta_rep_day1=theta_r[0, :][None, :],
                     theta_rep_day2=theta_r[1, :][None, :], 
                     k=4.0,
                     errorrates_stt = errorrates_stt,
                     errorrates_dtt = errorrates_dtt,
                     Q_init=jnp.array([[[0.2, 0, 0, 0.2]]*num_agents]))

    probs = jnp.squeeze(one_session(agent, exp_data))
    
    dfgh
    
    probabils = jnp.concatenate((errorrates_dtt*np.ones(probs.shape),(1-errorrates_dtt)*probs), axis = 1)

    print("Only for 1 participant")
    # not_observed = [i for i in range(len(exp_data['choices'])) if exp_data['choices'][i] < 0]
    "Watch out: Will always remove the 0th entry, Baby!"
    # not_observed = jnp.unique(jnp.arange(len(exp_data['choices'])) * jnp.squeeze((exp_data['choices'] < 0)))
    "Can I just use numpy here?"
    # not_observed = jnp.unique(jnp.arange(len(exp_data['choices']))[jnp.squeeze(exp_data['choices'] < 0)])
    
    # print("The shapes before")
    # print(probs.shape)
    
    # observed = jnp.delete(exp_data['bin_choices'], jnp.asarray(not_observed))
    # probabils = jnp.delete(probs, jnp.asarray(not_observed), axis = 0)
    
    # print("The shapes after")
    # print(observed.shape)
    # print(jnp.squeeze(probabils).shape)

    numpyro.sample('like', dist.Categorical(probs=jnp.squeeze(probabils)), obs=observed)

def one_session(agent, exp_data):
    """Run one entire session with all trials using the jax agent."""
    # self.prepare_sims()

    # num_agents = num_parts
    # The index of the block in the current experiment

    def one_trial(carry, matrices):
        day, trial, blocktype, current_choice, outcome = matrices
        Q, pppchoice, ppchoice, pchoice, seq_counter, rep, V = carry
        
        print("V[-1] shape, baby!")
        print(V[-1].shape)
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
    
    print("What about new block trials?")
    print("This might only be suitable for a single participant.")
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
    
    print("agent V[-1] shape")
    print(agent.V[-1].shape)
    key, probs = lax.scan(one_trial, carry, matrices)

    "Necessarily, probs still contains the values for the new block trials and single-target trials."

    return probs[0]

# def shorten_data(exp_data):
    
#     "Make sure that all values of exp_data have the same length"
#     lengths = [len(exp_data[key]) for key in exp_data.keys()]
#     assert(np.abs(np.diff(np.array(lengths))).sum() == 0)
    
#     for key in exp_data.keys():
#         exp_data[key] = exp_data[key][0:lengths[0]//3]
        
#     return exp_data

#%%

num_agents = 4

num_chains = 1
num_samples = 500

# _, exp_data = mj.simulation(num_agents = num_agents, 
#                             lr_day1 = jnp.array([[0.1]]),
#                             lr_day2 = jnp.array([[0.1]]),
#                             theta_Q_day1 = jnp.array([[2.]]),
#                             theta_Q_day2 = jnp.array([[2.]]),
#                             theta_rep_day1 = jnp.array([[3.]]),
#                             theta_rep_day2 = jnp.array([[3.]]))

_, exp_data = mj.simulation(num_agents = num_agents, errorrates_dtt = 0.2, errorrates_stt = 0.1) 
                            # sequence = [1,1,1,1],
                            # blockorder = [1,1,1,1])

"not_observed: errors (-2) as well as single-target trials and new block trials."
new_block_trials = jnp.nonzero(jnp.squeeze(jnp.asarray(exp_data['trialsequence'])[:,0] == -1))[0]
choices_wo_newblocktrials = jnp.delete(jnp.asarray(exp_data['choices']), new_block_trials, axis = 0)
trialseq_wo_newblocktrials = jnp.delete(jnp.asarray(exp_data['trialsequence']), new_block_trials, axis = 0)

num_stt = 5712
num_dtt = 1008

errorrates_stt = jnp.count_nonzero(((trialseq_wo_newblocktrials < 10) * choices_wo_newblocktrials) == -2, axis = 0) / num_stt
errorrates_dtt = jnp.count_nonzero(((trialseq_wo_newblocktrials > 10) * choices_wo_newblocktrials) == -2, axis = 0) / num_dtt

# "Not observed: new block trials and single target trials."
# not_observed = jnp.tile(jnp.arange(len(exp_data['choices'])),(num_agents,1)).T* jnp.squeeze(jnp.asarray(exp_data['trialsequence']) < 10)

"The rows where no participant saw a dual-target trial"
non_dtt_row_indices = np.where(np.all(np.array(exp_data['trialsequence']) < 10, axis=1))[0]

# "Unroll the not_observeds into one long stupid array"
# not_observed_unrolled = jnp.empty((0,), dtype=int)
# for ag in range(num_agents):
#     not_observed_unrolled = jnp.concatenate((not_observed_unrolled,
#                                              jnp.unique(not_observed[:, ag])))



num_trials = 6734
num_agents = 4
probs=np.random.rand(num_trials, num_agents ,2)

"Add 1s for the errors"
probs = np.concatenate((np.ones((num_trials, num_agents, 1)), probs), axis=2)

probs_stt_mask = (np.array(exp_data['trialsequence']) < 10)[..., None]
probs_dtt_mask = (np.array(exp_data['trialsequence']) > 10)[..., None]

"STT"
"Replace last dimension with the error rates for stt trials"
probs_new = probs.copy()
probs_new[:, :, -1] = np.where(probs_stt_mask, 
                           np.ones((num_trials, num_agents, 3))*np.tile(errorrates_stt, (num_trials, 1))[..., None], 
                           probs_stt_mask)[:, :, -1]

"Multiply "


#%%
kernel = NUTS(define_model, dense_mass=True)
mcmc = MCMC(kernel, 
            num_warmup=500, 
            num_samples=num_samples,
            num_chains=num_chains, 
            progress_bar=num_chains == 1)

rng_key = random.PRNGKey(1)
mcmc.run(rng_key, 
         exp_data=exp_data, 
         num_agents=num_agents,
         non_dtt_row_indices = non_dtt_row_indices,
         errorrates_stt = errorrates_stt,
         errorrates_dtt = errorrates_dtt)
mcmc.print_summary()


#%%
# group_data = utils.get_group_data()


