import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
import numpyro

# numpyro.set_platform('cpu')
# numpyro.set_host_device_count(4)

import ipdb
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from jax import numpy as jnp
from jax import random as jran
import numpy as np
import model_jax as mj


def define_model_groupinference(agent,
                non_dtt_row_indices,
                errorrates_stt,
                errorrates_dtt):
    
    """Define priors and likelihood."""
    observed = jnp.delete(agent.data['bin_choices_w_errors'], 
                          non_dtt_row_indices, 
                          axis = 0)
    
    a = numpyro.param('a', jnp.ones(agent.num_parameters), constraint=dist.constraints.positive)
    lam = numpyro.param('lam', jnp.ones(agent.num_parameters), constraint=dist.constraints.positive)
    tau = numpyro.sample('tau', dist.Gamma(a, a/lam).to_event(1)) # Why a/lam?
    
    sig = 1/jnp.sqrt(tau) # Gauss sigma

    # each model parameter has a hyperprior defining group level mean
    # in the form of a Normal distribution
    m = numpyro.param('m', jnp.zeros(agent.num_parameters))
    s = numpyro.param('s', jnp.ones(agent.num_parameters), constraint=dist.constraints.positive)
    mu = numpyro.sample('mu', dist.Normal(m, s*sig).to_event(1)) # Gauss mu, wieso s*sig?

    with numpyro.plate('subject', agent.num_agents):
        # draw parameters from Normal and transform (for numeric trick reasons)
        base_dist = dist.Normal(0., 1.).expand_by([agent.num_parameters]).to_event(1)
        transform = dist.transforms.AffineTransform(mu, sig)
        # print("CHECKPOINT BRAVO")
        locs = numpyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))

        "locs is either of shape [n_participants, n_parameters] or of shape [n_particles, n_participants, n_parameters]"
        if locs.ndim == 2:
            locs = locs[None, :]

        param_dict = agent.locs_to_pars(locs)
                
        probs = jnp.squeeze(agent.jitted_one_session(lr_day1 = param_dict['lr_day1'],
                                                     lr_day2 = param_dict['lr_day2'],
                                                     theta_Q_day1 = param_dict['theta_Q_day1'],
                                                     theta_Q_day2 = param_dict['theta_Q_day2'],
                                                     theta_rep_day1 = param_dict['theta_rep_day1'],
                                                     theta_rep_day2 = param_dict['theta_rep_day2']))
        
        # dfgh
        "Remove new block trials"
        probabils = jnp.delete(probs, non_dtt_row_indices, axis = 0)

        with numpyro.plate('time', probabils.shape[0]):
            numpyro.sample('like',
                            dist.Categorical(probs=probabils), 
                            obs=observed)

def define_model_singleinference(agent,
                non_dtt_row_indices,
                errorrates_stt,
                errorrates_dtt):
    
    """Define priors and likelihood."""
    observed = jnp.delete(agent.data['bin_choices_w_errors'], 
                          non_dtt_row_indices, 
                          axis = 0)
    
    with numpyro.plate('subject', 1):
        lr_day1 = numpyro.sample('lr_day1', dist.Beta(2, 3))
        lr_day2 = numpyro.sample('lr_day2', dist.Beta(2, 3))
        
        theta_Q_day1 = numpyro.sample('theta_Q_day1', dist.HalfNormal(8.))#.expand([2]))
        theta_Q_day2 = numpyro.sample('theta_Q_day2', dist.HalfNormal(8.))#.expand([2]))
    
        theta_rep_day1 = numpyro.sample('theta_rep_day1', dist.HalfNormal(8.))#.expand([2]))
        theta_rep_day2 = numpyro.sample('theta_rep_day2', dist.HalfNormal(8.))#.expand([2]))
        
        probs = jnp.squeeze(agent.jitted_one_session(lr_day1 = jnp.array([lr_day1]),
                                                     lr_day2 = jnp.array([lr_day2]),
                                                     theta_Q_day1 = jnp.array([theta_Q_day1]),
                                                     theta_Q_day2 = jnp.array([theta_Q_day2]),
                                                     theta_rep_day1 = jnp.array([theta_rep_day1]),
                                                     theta_rep_day2 = jnp.array([theta_rep_day2])))
        
        "Remove new block trials"
        probabils = jnp.delete(probs, non_dtt_row_indices, axis = 0)
        # dfgh
    
        # dfgh
        with numpyro.plate('timesteps', probabils.shape[0]):
            numpyro.sample('like',
                            dist.Categorical(probs=probabils), 
                            obs=observed)

def perform_grouplevel_inference(agent, num_samples, num_warmup):
    
    num_chains = 1
    num_samples = num_samples
    num_warmup = num_warmup
    
    new_block_trials = jnp.nonzero(jnp.squeeze(jnp.asarray(agent.data['trialsequence'])[:,0] == -1))[0]
    choices_wo_newblocktrials = jnp.delete(jnp.asarray(agent.data['choices']), new_block_trials, axis = 0)
    trialseq_wo_newblocktrials = jnp.delete(jnp.asarray(agent.data['trialsequence']), new_block_trials, axis = 0)
    
    num_stt = 5712
    num_dtt = 1008
    
    "Compute error rates"
    ER_stt = jnp.count_nonzero(((trialseq_wo_newblocktrials < 10) * choices_wo_newblocktrials) == -2, axis = 0) / num_stt
    ER_dtt = jnp.count_nonzero(((trialseq_wo_newblocktrials > 10) * choices_wo_newblocktrials) == -2, axis = 0) / num_dtt
    
    "Find rows where no participant saw a dtt"
    non_dtt_row_indices = jnp.where(jnp.all(jnp.array(agent.data['trialsequence']) < 10, axis=1))[0]
    
    kernel = NUTS(define_model_groupinference, dense_mass=True)
        
    mcmc = MCMC(kernel, 
                num_warmup=num_warmup, 
                num_samples=num_samples,
                num_chains=num_chains, 
                progress_bar=True)
    
    rng_key = jran.PRNGKey(np.random.randint(10_000))
    
    mcmc.run(rng_key, 
              agent = agent,
              non_dtt_row_indices = non_dtt_row_indices,
              errorrates_stt = jnp.asarray([ER_stt]),
              errorrates_dtt = jnp.asarray([ER_dtt]))
        
    return mcmc


def perform_firstlevel_inference(agent, num_samples, num_warmup):
    
    num_chains = 1
    
    new_block_trials = jnp.nonzero(jnp.squeeze(jnp.asarray(agent.data['trialsequence'])[:,0] == -1))[0]
    choices_wo_newblocktrials = jnp.delete(jnp.asarray(agent.data['choices']), new_block_trials, axis = 0)
    trialseq_wo_newblocktrials = jnp.delete(jnp.asarray(agent.data['trialsequence']), new_block_trials, axis = 0)
    
    num_stt = 5712
    num_dtt = 1008
    
    "Compute error rates"
    ER_stt = jnp.count_nonzero(((trialseq_wo_newblocktrials < 10) * choices_wo_newblocktrials) == -2, axis = 0) / num_stt
    ER_dtt = jnp.count_nonzero(((trialseq_wo_newblocktrials > 10) * choices_wo_newblocktrials) == -2, axis = 0) / num_dtt
    
    "Find rows where no participant saw a dtt"
    non_dtt_row_indices = jnp.where(jnp.all(jnp.array(agent.data['trialsequence']) < 10, axis=1))[0]
    
    kernel = NUTS(define_model_singleinference, dense_mass=True)
    
    mcmc = MCMC(kernel, 
                num_warmup=num_warmup, 
                num_samples=num_samples,
                num_chains=num_chains, 
                progress_bar=True)
    
    rng_key = jran.PRNGKey(np.random.randint(10_000))
    
    
    mcmc.run(rng_key, 
              agent = agent,
              non_dtt_row_indices = non_dtt_row_indices,
              errorrates_stt = jnp.asarray([ER_stt]),
              errorrates_dtt = jnp.asarray([ER_dtt]))
        
    return mcmc

#%%
# group_data = utils.get_group_data()


#%%

