import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.utils.metrics.stats import Stats

from ray.rllib.utils.typing import ResultDict
from typing import Tuple
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EVALUATION_RESULTS,
)

'''
    In this method, we want to run some number of episodes each against each of the heuristic policies, and report the win/loss/draw rate for each.
     - Get the number of evaluation episodes, collect the episodes, switch the heuristic policy every 1/K of the way through. I think we can aggregate directly.
     - The first step is to do this without switching opponents.
'''

def custom_eval_function(
    algorithm: Algorithm,
    eval_workers: EnvRunnerGroup,
) -> Tuple[ResultDict, int, int]:
    env_runner_metrics = []
    sampled_episodes = []
    # For demonstration purposes, run through some number of evaluation rounds within this one call. Note that this function is called once per training iteration (`Algorithm.train()` call) OR once per `Algorithm.evaluate()`
    heuristics = algorithm.config.evaluation_config['heuristics']
    ensemble = algorithm.config.evaluation_config['ensemble']
    roles = algorithm.config.evaluation_config['roles']
    to_sample = int(algorithm.config.evaluation_duration / len(heuristics) / len(roles))
    for h in heuristics:
      for role in roles: # I think this is the neatest way to do both
        def agent_to_module_mapping_fn(agent_id, episode, **kwargs):
            return np.random.choice(ensemble) if (agent_id==role) else h
        def _add(_env_runner):
            _env_runner.config.multi_agent(
                policy_mapping_fn=agent_to_module_mapping_fn
            )
            #return MultiRLModuleSpec.from_module(_env_runner.module)
        eval_workers.foreach_env_runner(_add)
        #
        metrics_h = []
        episodes_h = []
        for i in range(to_sample):
            # Sample episodes from the EnvRunners AND have them return only the thus
            # collected metrics.
            episodes_and_metrics_all_env_runners = eval_workers.foreach_env_runner(
                # Return only the metrics, NOT the sampled episodes (we don't need them
                # anymore).
                func=lambda worker: (worker.sample(), worker.get_metrics()),
                local_env_runner=(eval_workers.num_remote_workers()==0),
            )
            episodes_h.extend(
                eps
                for eps_and_mtrcs in episodes_and_metrics_all_env_runners
                for eps in eps_and_mtrcs[0]
            )
            metrics_h.extend(
                eps_and_mtrcs[1] for eps_and_mtrcs in episodes_and_metrics_all_env_runners
            )
        # For each heuristic, go through its sampled set of episodes and compute win rate
        for (ep, m) in zip(episodes_h, metrics_h):
          r = ep.agent_episodes[role].get_return()
          m['Win Rates'] = {}
          m['Win Rates'][role] = {h:{ # Replace $heuristic with the heuristic, try mean reduction
              'Win': Stats(init_values=(r==1), reduce="mean"),
              'Draw': Stats(init_values=(r==0), reduce="mean"),
              'Loss': Stats(init_values=(r==-1), reduce="mean"),
          }}
        env_runner_metrics.extend(metrics_h)
        sampled_episodes.extend(episodes_h)
    # You can compute metrics from the episodes manually, or use the Algorithm's
    # convenient MetricsLogger to store all evaluation metrics inside the main
    # algo.
    algorithm.metrics.aggregate(
        env_runner_metrics, key=(EVALUATION_RESULTS, ENV_RUNNER_RESULTS)
    )
    eval_results = algorithm.metrics.peek((EVALUATION_RESULTS, ENV_RUNNER_RESULTS))
    # Alternatively, you could manually reduce over the n returned `env_runner_metrics`
    # dicts, but this would be much harder as you might not know, which metrics
    # to sum up, which ones to average over, etc..

    # Compute env and agent steps from sampled episodes.
    env_steps = sum(eps.env_steps() for eps in sampled_episodes)
    agent_steps = sum(eps.agent_steps() for eps in sampled_episodes)
    return eval_results, env_steps, agent_steps