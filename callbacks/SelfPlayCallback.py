from collections import defaultdict

import numpy as np

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS

from ray.rllib.core import (
    COMPONENT_ENV_RUNNER,
    COMPONENT_EVAL_ENV_RUNNER,
    COMPONENT_LEARNER,
    COMPONENT_LEARNER_GROUP,
)

# From our code
from algo.constants import SHARED_CRITIC_ID

def create_atm_fn(frozen_policies, exploiter_rate, self_play_rate, pfsp_weights=None): # loss rate is an array corresponding to main_vi
  def atm_fn(agent_id, episode, **kwargs):
    # If main is assigned to the target agent this episode, return main
    if (hash(episode.id_) % 2 == 0) != (agent_id=='X'):
      return "main"
    # Else, select an opponent
    rand = np.random.rand()
    if (rand < self_play_rate):
      return "main"
    elif (rand < self_play_rate + exploiter_rate):
      return "main_exploiter"
    else:
      frozen_v = np.random.choice(list(range(frozen_policies)),p=pfsp_weights)
      return f'main_v{frozen_v}'
  return atm_fn

class SelfPlayCallback(RLlibCallback):
    def __init__(self, win_rate_threshold, _lambda=0.1, max_league_size=32,self_play=0,exploiter=0, window_size=10000):
        super().__init__()
        # 0=main_v0, 1=main_v1, 2=2nd main policy snapshot, etc..
        self.frozen_policies = 1
        self.win_rate_threshold = win_rate_threshold
        self._lambda = _lambda
        self.max_league_size = max_league_size
        self.window_size = window_size
        self.self_play_rate=self_play
        self.exploiter_rate=exploiter
        # Report the matchup counters (who played against whom?).
        self._matching_stats = defaultdict(int)
        # Hacky fix for the new agent WR bug
        self.just_added = False

    def on_episode_end(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ) -> None:
        # Compute the win rate for this episode and log it with a window of 100.
        # Exclude self-play, since it'll always be 50/50
        if (episode.module_for('X')!=episode.module_for('O')):
          main_agent = 'X' if episode.module_for('X') == "main" else 'O' # agent that our main policy was
          opposing_agent = episode.module_for('O') if main_agent=='X' else episode.module_for('X')
          self._matching_stats[("main", opposing_agent)] += 1
          if 'exploiter' in opposing_agent:
            opposing_agent = 'exploiter'
          else:
            opposing_agent = opposing_agent.split('_v')[-1]
          rewards = episode.get_rewards()
          assert main_agent in rewards
          main_won = rewards[main_agent][-1] == 1.0
          main_lost = rewards[main_agent][-1] == -1.0
          metrics_logger.log_value(
              f"win_rate_{opposing_agent}",
              main_won,
              window=self.window_size,
          )
          metrics_logger.log_value(
              f"loss_rate_{opposing_agent}",
              main_lost,
              window=self.window_size,
          )
          
    def get_atm_fn(self, algorithm, loss_rates):
        if (loss_rates.sum() == 0):
          base_probs = loss_rates + 1 / len(loss_rates)
        else:
          base_probs = loss_rates / loss_rates.sum()
          # Divide up our lambda among the agents that are getting used less than it
          z = base_probs<self._lambda
          if (self.just_added): # Don't add p to the just-added one before it propagates to workers.
              z[-1] = False
          if (z.any()):
              base_probs *= (1.0-self._lambda)
              base_probs += z / z.sum() * self._lambda
        print(f"Updating ATM fn: {base_probs}")
        # Reweight and (if applicable) add to agent randomizer
        # Set new mapping function
        return create_atm_fn(self.frozen_policies, self.exploiter_rate, self.self_play_rate, base_probs)

    def update_atm_fn(self, algorithm, loss_rates):
        #
        agent_to_module_mapping_fn = self.get_atm_fn(algorithm, loss_rates)
        algorithm.config._is_frozen = False
        algorithm.config.multi_agent(policy_mapping_fn=agent_to_module_mapping_fn)
        algorithm.config.freeze()
        # Add to (training) EnvRunners.
        def _add(_env_runner, _module_spec=None):
            _env_runner.config.multi_agent(
                policy_mapping_fn=agent_to_module_mapping_fn,
            )
            return MultiRLModuleSpec.from_module(_env_runner.module)
        algorithm.env_runner_group.foreach_env_runner(_add)

    def on_train_result(self, *, algorithm, metrics_logger=None, result, **kwargs):
        print(f"Iter={algorithm.iteration}:")
        print(f"Matchups: {dict(self._matching_stats)}")
        if (self.exploiter_rate+self.self_play_rate == 1.0):
          return # No PFSP means skip the rest.
        f_wr = f_lr = 0
        worst_ratio = 1 # worst ratio must exceed threshold
        loss_rates = np.array([(result[ENV_RUNNER_RESULTS][f"loss_rate_{i}"] if f"loss_rate_{i}" in result[ENV_RUNNER_RESULTS] else 0) for i in range(self.frozen_policies)])
        for i in range(self.frozen_policies):
          win_rate = result[ENV_RUNNER_RESULTS][f"win_rate_{i}"] if f"win_rate_{i}" in result[ENV_RUNNER_RESULTS] else 0
          loss_rate = loss_rates[i]
          sum_rates = (win_rate + loss_rate)
          ratio = win_rate / sum_rates if sum_rates > 0 else 1
          print(f"Opponent {i}: win-rate={win_rate:.2f} loss-rate={loss_rate:.2f} ... ", end="" if i == self.frozen_policies else '\n')
          f_wr, f_lr = win_rate, loss_rate
          if (ratio < worst_ratio):
            worst_ratio = ratio
        # Exploiter
        if (self.exploiter_rate > 0):
          win_rate = result[ENV_RUNNER_RESULTS][f"win_rate_exploiter"]
          loss_rate = result[ENV_RUNNER_RESULTS][f"loss_rate_exploiter"]
          print(f"Exploiter: win-rate={win_rate:.2f} loss-rate={loss_rate:.2f} ... ", end="" if i == self.frozen_policies else '\n')
        # If win rate is good versus the most recent opponent -> Snapshot current policy and play against
        # it next, keeping the snapshot fixed and only improving the "main"
        # policy.
        if (self.frozen_policies+2 == self.max_league_size):
            print("maximum league size has been reached.")
        elif (self.just_added):
            self.just_added = False
            print("just added new agent; allowing an epoch to propagate.")
        elif f_wr > 0 and worst_ratio > self.win_rate_threshold:
            self.just_added = True
            new_module_id = f"main_v{self.frozen_policies}"
            self.frozen_policies += 1
            print(f"adding new opponent to the mix ({new_module_id}).")
            loss_rates = np.append(loss_rates, [0.0]) # b/c of prop. issue
            main_module = algorithm.get_module("main")
            algorithm.add_module(
                module_id=new_module_id,
                module_spec=RLModuleSpec.from_module(main_module), # Copy main module specs
                new_agent_to_module_mapping_fn=self.get_atm_fn(algorithm, loss_rates),
            )
            module_updates = {new_module_id: main_module.get_state(),}
            # update shared critic, syncing weights (get_module only looks at envrunners, can't use)
            sc = algorithm.learner_group._learner._module[SHARED_CRITIC_ID]
            if (sc.identity_aug):
                sc.new_agent_embedding(-(self.max_league_size-1-self.frozen_policies))
                module_updates[SHARED_CRITIC_ID] = sc.get_state()
            # Syncs weights across everything (wait, why does it just say 'learner group'?)
            algorithm.set_state(
                {
                    COMPONENT_LEARNER_GROUP: {
                        "learner": {
                            "rl_module": module_updates
                        }
                    },
                }
            )
        else:
            print("not good enough; will keep learning ...")

        if (not self.just_added):
            # Update mapping function, reweighting and adding new module if needed
            self.update_atm_fn(algorithm, loss_rates)

        # +2 = main + exploiter
        result["league_size"] = self.frozen_policies + 2