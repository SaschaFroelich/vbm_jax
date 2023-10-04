import numpyro
from numpyro.infer import MCMC, NUTS, SA
import numpyro.distributions as dist
from jax import numpy as jnp
from jax import random, lax
import numpy as np

import model_jax as mj

def define_model(exp_data):
    """Define priors and likelihood."""
    lrs = numpyro.sample('lrs', dist.Beta(3, 2).expand(2))
    theta_q = numpyro.sample('theta_q', dist.HalfNormal(5).expand(2))
    theta_r = numpyro.sample('theta_q', dist.HalfNormal(5).expand(2))

    agent = mj.Vbm_B(lr_day1=lrs[0], lr_day2=lrs[1], theta_Q_day1=theta_q[0],
                     theta_Q_day2=theta_q[1], theta_rep_day1=theta_r[0],
                     theta_rep_day2=theta_r[1], k=4.0,
                     Q_init=[[[0.2, 0, 0, 0.2]]])

    probs = one_session(agent, exp_data)

    numpyro.sample('like', dist.Categorical(probs=probs), obs=exp_data.choices)

def one_session(agent, exp_data, num_parts):
    """Run one entire session with all trials using the jax agent."""
    self.prepare_sims()

    num_agents = num_parts
    # The index of the block in the current experiment

    def one_trial(dummy, matrices):
        day, trial, blocktype, current_choice, outcome = matrices
        probs = agent.compute_probs(trial, day)
        agent.update(jnp.asarray(current_choice),
                     jnp.asarray(outcome), blocktype,
                     day=day, trial=trial)
        outtie = [probs]
        return dummy, outtie
    
    days = (np.array(self.data['blockidx']) > 5) + 1
    trials = np.squeeze(self.data["trialsequence"])
    lin_blocktype = jnp.hstack([-jnp.ones((14, 1), dtype=int),
                                blocktype.astype(int)])
    lin_blocktype = jnp.repeat(lin_blocktype.reshape(-1)[None, ...],
                               num_agents, axis=0)
    trials = jnp.repeat(trials[None, ...], num_agents, axis=0)
    matrices = [days, trials.T, lin_blocktype.T, choices, outcomes]
    dummy = 1337
    key, probs = lax.scan(one_trial, dummy, matrices)
    return probs

num_chains = 8
num_samples = 1_000_000

kernel = NUTS(define_model, dense_mass=True)
mcmc = MCMC(kernel, num_warmup=500, num_samples=num_samples,
            num_chains=num_chains, progress_bar=num_chains == 1)
rng_key = random.PRNGKey(1)
mcmc.run(rng_key, exp_data=exp_data)
mcmc.print_summary()