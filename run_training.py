# @title config_league
import numpy as np
import functools

from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule

from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)

from environments.TicTacToe import *

from classes.heuristics import RandHeuristicRLM, BlockWinHeuristicRLM, PerfectHeuristicRLM

from callbacks.custom_eval_function import custom_eval_function
from callbacks.SelfPlayCallback import SelfPlayCallback

from algo.modules.CMAPPOActionMaskingTorchRLModule import CMAPPOActionMaskingTorchRLModule
from algo.modules.critic.ActionMaskingSharedCriticTorchRLModule import *
from algo.modules.critic.SharedCriticCatalog import (
    LOGITS,
    ACTIONS,
    LOGITS_AND_ACTIONS,
)
from algo.CMAPPOConfig import CMAPPOConfig
from algo.constants import SHARED_CRITIC_ID
from algo.CMAPPOTorchLearner import CMAPPOTorchLearner

from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
)
from classes.run_tune_training import run_tune_training
from ray.rllib.utils.metrics import (
    TRAINING_ITERATION_TIMER,
)

parser = add_rllib_example_script_args(default_iters=100)
parser.set_defaults(
    num_env_runners=0,
    verbose=1
)
parser.add_argument("--lr", type=float, default=1e-4) 
parser.add_argument('--critic-fcnet', nargs='+', type=int, default=[256,256]) # Head architecture
parser.add_argument("--batch-size", type=int, default=4096)
parser.add_argument("--minibatch-size", type=int, default=128)
parser.add_argument("--restore-checkpoint", type=str)

parser.add_argument("--self-aug", choices=[LOGITS, ACTIONS, LOGITS_AND_ACTIONS])
parser.add_argument("--other-aug", choices=[LOGITS, ACTIONS, LOGITS_AND_ACTIONS])

args = parser.parse_args()


specs = {
    "rand": RLModuleSpec(
        module_class=RandHeuristicRLM,
    ),
    "block_win": RLModuleSpec(
        module_class=BlockWinHeuristicRLM,
    ),
    "perfect": RLModuleSpec(
        module_class=PerfectHeuristicRLM,
    ),
}
heuristics = list(specs.keys())

for n in ['main', 'main_v0']: # default frozen policy, and first learned policy
    p = n
    specs[p] =  RLModuleSpec(
        module_class=CMAPPOActionMaskingTorchRLModule,
        model_config={
            "head_fcnet_hiddens": (64,64),
        }
    )

# Shared critic. Might seem moot when training only one agent, but we can use it to neatly augment critic outputs with logits, and maybe use frozen opponents' results to train the critic too.
single_agent_env = TicTacToe()
specs[SHARED_CRITIC_ID] = RLModuleSpec(
        module_class=ActionMaskingSharedCriticTorchRLModule,
        observation_space=single_agent_env.observation_spaces['X'],
        action_space=single_agent_env.action_spaces['X'],
        learner_only=True, # Only build on learner
        model_config={
            "head_fcnet_hiddens": tuple(args.critic_fcnet),
            "self_aug": args.self_aug,
            "other_aug": args.other_aug,
            "logits_size": single_agent_env.action_spaces['X'].n, 
        },
    )

# League stuff
win_rate_threshold = 0.95 # wins / wins+losses, wins > 2
def agent_to_module_mapping_fn(agent_id, episode, **kwargs):
    # agent_id = [0|1] -> module depends on episode ID
    # This way, we make sure that both modules sometimes play agent0
    # (start player) and sometimes agent1 (player to move 2nd).
    return "main" if ((hash(episode.id_) % 2 == 0) != (agent_id=='X')) else "main_v0"

config = (
    CMAPPOConfig()
    .environment(TicTacToe, env_config={})
    .callbacks( # set up our league
        functools.partial(SelfPlayCallback,
            win_rate_threshold=win_rate_threshold,
            _lambda=0.1 # Total probability to allocate to agents with zero WR vs main
        )
    )
    .env_runners(
        num_env_runners=args.num_env_runners,
        batch_mode="complete_episodes", # use_logits needs this to work.
    )
    .evaluation(
        custom_evaluation_function=custom_eval_function,
        evaluation_interval=10, # every K training steps
        evaluation_duration=100,
        evaluation_config={
            'heuristics': heuristics,
            'ensemble': ['main'], # policies to use when calculating winrates
        }
    )
    .multi_agent(
          policies=['main','main_v0', SHARED_CRITIC_ID]+heuristics,
          policy_mapping_fn=agent_to_module_mapping_fn,
          # Only the learned policy should be trained.
          policies_to_train=['main', SHARED_CRITIC_ID],
      )
    .training(
        learner_class=CMAPPOTorchLearner,
        lr=args.lr * args.num_env_runners**0.5, # The general rule for scaling up
        train_batch_size_per_learner=args.batch_size,
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs=specs
        ),
    )
)

'''
algo = config.build_algo()

from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
)

num_iters = 100

for i in range(num_iters):
  results = algo.train()
  if ENV_RUNNER_RESULTS in results:
      mean_return = results[ENV_RUNNER_RESULTS].get(
          'agent_episode_returns_mean', np.nan
      )
      mean_return = [(k, f'{v:.2f}') for k, v in mean_return.items()]
      print(f"iter={i+1} R={mean_return}")
  if (
      algo.config.evaluation_interval is not None
      and (i+1)%algo.config.evaluation_interval == 0):
      wrs = results[EVALUATION_RESULTS][ENV_RUNNER_RESULTS].get("Win Rates")
      for k in sorted(wrs.keys()):
        print(f"\t{k}")
        for k2 in ['Win','Draw','Loss']:
          print(f"\t\t{k2}: {wrs[k][k2]:.2f}") 
          
'''


stop = {
    TRAINING_ITERATION_TIMER: args.stop_iters,
}

# Load policy if applicable
if (args.restore_checkpoint):
    print(f"Restoring checkpoint: {args.restore_checkpoint}")
    assignRestoreCallback(args.restore_checkpoint, config)

# Run the experiment.
run_tune_training(config,args,stop=stop) #'''