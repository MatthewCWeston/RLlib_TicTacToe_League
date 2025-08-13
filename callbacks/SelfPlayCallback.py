# @title SelfPlayCallback
from collections import defaultdict

import numpy as np

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS



class SelfPlayCallback(RLlibCallback):
    def __init__(self, win_rate_threshold, _lambda):
        super().__init__()
        # 0=main_v0, 1=main_v1, 2=2nd main policy snapshot, etc..
        self.current_opponent = 0
        self.win_rate_threshold = win_rate_threshold
        self._lambda = _lambda
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
        main_agent = 'X' if episode.module_for('X') == "main" else 'O' # agent that our main policy was
        opposing_agent = episode.module_for('X') if main_agent=='O' else episode.module_for('O')
        opposing_agent = opposing_agent.split('_v')[-1]
        rewards = episode.get_rewards()
        assert main_agent in rewards
        main_won = rewards[main_agent][-1] == 1.0
        main_lost = rewards[main_agent][-1] == -1.0
        metrics_logger.log_value(
            f"win_rate_{opposing_agent}",
            main_won,
            window=1000,
        )
        metrics_logger.log_value(
            f"loss_rate_{opposing_agent}",
            main_lost,
            window=1000,
        )

    def update_atm_fn(self, algorithm, loss_rates):
        #
        base_probs = loss_rates / loss_rates.sum()
        # Divide up 10% among the zeroes
        z = base_probs==0
        if (z.any()):
            base_probs *= (1.0-self._lambda)
            base_probs += z / z.sum() * self._lambda
        # Reserve a one percent chance for 
        print(f"Updating ATM fn: {base_probs}")
        # Reweight and (if applicable) add to agent randomizer
        def agent_to_module_mapping_fn(agent_id, episode, **kwargs):
            opponent = "main_v{}".format(
                np.random.choice(list(range(self.current_opponent + 1)),p=base_probs)
            )
            if ((hash(episode.id_) % 2 == 0) != (agent_id=='X')):
                self._matching_stats[("main", opponent)] += 1
                return "main"
            else:
                return opponent
        # Set new mapping function
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
        f_wr = f_lr = 0
        worst_ratio = 1 # worst ratio must exceed threshold
        loss_rates = np.array([result[ENV_RUNNER_RESULTS][f"loss_rate_{i}"] for i in range(self.current_opponent+1)])
        for i in range(self.current_opponent+1):
          win_rate = result[ENV_RUNNER_RESULTS][f"win_rate_{i}"]
          loss_rate = loss_rates[i]
          sum_rates = (win_rate + loss_rate)
          ratio = win_rate / sum_rates if sum_rates > 0 else 1
          print(f"Opponent {i}: win-rate={win_rate:.2f} loss-rate={loss_rate:.2f} ... ", end="" if i == self.current_opponent else '\n')
          f_wr, f_lr = win_rate, loss_rate
          if (ratio < worst_ratio):
            worst_ratio = ratio

        # If win rate is good versus the most recent opponent -> Snapshot current policy and play against
        # it next, keeping the snapshot fixed and only improving the "main"
        # policy.
        if (self.just_added):
            self.just_added = False
            print("just added new agent; allowing an epoch to propagate.")
        elif f_wr > 0 and worst_ratio > self.win_rate_threshold:
            self.current_opponent += 1
            new_module_id = f"main_v{self.current_opponent}"
            print(f"adding new opponent to the mix ({new_module_id}).")

            # Reset stored values
            for i in range(self.current_opponent):
                metrics_logger.set_value(
                    f"win_rate_{i}",
                    0,
                    window=1000,
                )
                metrics_logger.set_value(
                    f"loss_rate_{i}",
                    0,
                    window=1000,
                )
            loss_rates = np.append(loss_rates, [0.0]) # b/c of prop. issue
            main_module = algorithm.get_module("main")
            algorithm.add_module(
                module_id=new_module_id,
                module_spec=RLModuleSpec.from_module(main_module), # Copy main module specs
            )
            algorithm.set_state(
                {
                    "learner_group": {
                        "learner": {
                            "rl_module": {
                                new_module_id: main_module.get_state(),
                            }
                        }
                    }
                }
            )
            self.just_added = True
        else:
            print("not good enough; will keep learning ...")

        # Update mapping function, reweighting and adding new module if needed
        self.update_atm_fn(algorithm, loss_rates)

        # +2 = main + random
        result["league_size"] = self.current_opponent + 2