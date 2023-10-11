import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import numpyro

numpyro.set_platform('cpu')
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


def define_model(agent,
                exp_data,
                non_dtt_row_indices,
                errorrates_stt,
                errorrates_dtt):
    
    """Define priors and likelihood."""
    observed = jnp.delete(exp_data['bin_choices_w_errors'], non_dtt_row_indices, axis = 0)
    
    a = numpyro.param('a', jnp.ones(agent.num_parameters), constraint=dist.constraints.positive)
    lam = numpyro.param('lam', jnp.ones(agent.num_parameters), constraint=dist.constraints.positive)
    tau = numpyro.sample('tau', dist.Gamma(a, a/lam).to_event(1)) # Why a/lam?
    
    sig = 1/jnp.sqrt(tau) # Gaus sigma

    # each model parameter has a hyperprior defining group level mean
    # in the form of a Normal distribution
    m = numpyro.param('m', jnp.zeros(agent.num_parameters))
    s = numpyro.param('s', jnp.ones(agent.num_parameters), constraint=dist.constraints.positive)
    mu = numpyro.sample('mu', dist.Normal(m, s*sig).to_event(1)) # Gauss mu, wieso s*sig?

    with numpyro.plate('subject', agent.num_agents) as ind:
        
        # draw parameters from Normal and transform (for numeric trick reasons)
        base_dist = dist.Normal(0., 1.).expand_by([agent.num_parameters]).to_event(1)
        transform = dist.transforms.AffineTransform(mu, sig)
        # print("CHECKPOINT BRAVO")
        locs = numpyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))

        "locs is either of shape [n_participants, n_parameters] or of shape [n_particles, n_participants, n_parameters]"
        if locs.ndim == 2:
            locs = locs[None, :]
            
        agent.reset(locs)
        
        num_particles = locs.shape[0]
        
        probs = jnp.squeeze(one_session(agent, exp_data))
        "Remove new block trials"
        probabils = jnp.delete(probs, non_dtt_row_indices, axis = 0)
        with numpyro.plate('timesteps', probabils.shape[0]):
            numpyro.sample('like',
                            dist.Categorical(probs=probabils), 
                            obs=observed)

def one_session(agent, exp_data):
    """Run one entire session with all trials using the jax agent."""
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

def perform_inference(agent, 
                      exp_data):
    
    num_agents = len(exp_data['trialsequence'][0])
    num_chains = 1
    num_samples = 500
    num_warmup = 500
    
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
                num_warmup=num_warmup, 
                num_samples=num_samples,
                num_chains=num_chains, 
                progress_bar=True)
    
    rng_key = random.PRNGKey(1)
    mcmc.run(rng_key, 
             agent = agent,
             exp_data = exp_data, 
             non_dtt_row_indices = non_dtt_row_indices,
             errorrates_stt = jnp.asarray([ER_stt]),
             errorrates_dtt = jnp.asarray([ER_dtt]))
    mcmc.print_summary()
    
    return mcmc

#%%
# group_data = utils.get_group_data()


