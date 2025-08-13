from typing import Any, List, Dict

import numpy as np
import torch
from collections import defaultdict

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.common.numpy_to_tensor import NumpyToTensor
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.postprocessing.value_predictions import compute_value_targets
from ray.rllib.utils.postprocessing.zero_padding import (
    split_and_zero_pad_n_episodes,
    unpad_data_if_necessary,
)
from ray.rllib.utils.typing import EpisodeType

# From our code
from algo.constants import SHARED_CRITIC_ID, AGENT_LOGITS, OBSERVATIONS
from algo.modules.critic.SharedCriticCatalog import (
    LOGITS,
    ACTIONS,
    LOGITS_AND_ACTIONS,
)


class CMAPPOGAEConnector(ConnectorV2):
    def __init__(
        self,
        input_observation_space=None,
        input_action_space=None,
        *,
        gamma,
        lambda_,
    ):
        super().__init__(input_observation_space, input_action_space)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.aug_fn_dict = {
            LOGITS: self.get_logits,
            ACTIONS: self.get_actions,
            LOGITS_AND_ACTIONS: self.get_logits_and_actions
        }
        # Internal numpy-to-tensor connector to translate GAE results (advantages and
        # vf targets) into tensors.
        self._numpy_to_tensor_connector = None

    def get_logits(self, batch, mid, s, l):
      return batch[mid][Columns.ACTION_DIST_INPUTS][s:s+l]

    def get_actions(self, batch, mid, s, l):
      a_dim = batch[mid][Columns.ACTION_DIST_INPUTS].shape[1]
      return torch.nn.functional.one_hot(batch[mid][Columns.ACTIONS][s:s+l].long(), a_dim)

    def get_logits_and_actions(self, batch, mid, s, l):
      logits = self.get_logits(batch, mid, s, l)
      actions = self.get_actions(batch, mid, s, l)
      return torch.cat((actions, logits), dim=1)

    def augment_critic(self, batch, meps, aug_fn_self, aug_fn_othr, aug_size):
      for aid in batch:
        # Initialize next two action logit sets
        b_lgts = batch[aid][Columns.ACTION_DIST_INPUTS]
        batch[aid][AGENT_LOGITS] = torch.zeros((b_lgts.shape[0], aug_size)).to(b_lgts.device) # Create logit predictions
      start_indices = defaultdict(lambda: 0) # where to start in each agent's batch, when populating next action logits for critic
      for mep in meps:
        x_ep, o_ep = mep.agent_episodes['X'], mep.agent_episodes['O']
        assert x_ep.agent_id == 'X' # These episodes complete a game.
        assert o_ep.agent_id == 'O'
        assert x_ep.multi_agent_episode_id == o_ep.multi_agent_episode_id
        x_mid, o_mid = x_ep.module_id, o_ep.module_id
        x_l, o_l = len(x_ep), len(o_ep)
        # Start indices. We alternate because we might have the same module.
        x_s = start_indices[x_mid]
        start_indices[x_mid]+=x_l
        o_s = start_indices[o_mid]
        start_indices[o_mid]+=o_l
        # Grab aug vectors
        x_aug_self, o_aug_self = aug_fn_self(batch, x_mid, x_s, x_l), aug_fn_self(batch, o_mid, o_s, o_l)
        x_aug_othr, o_aug_othr = aug_fn_othr(batch, x_mid, x_s, x_l), aug_fn_othr(batch, o_mid, o_s, o_l)
        # Double-check that we have the right swath by comparing observations. Can comment out once we're sure.
        x_ep_obs = np.stack([o[OBSERVATIONS] for o in x_ep.observations][:x_l])
        tmp = np.array(batch[x_mid][Columns.OBS][OBSERVATIONS][x_s:x_s+x_l].cpu())
        assert np.abs(tmp-x_ep_obs).sum() == 0
        o_ep_obs = np.stack([o[OBSERVATIONS] for o in o_ep.observations][:o_l])
        tmp = np.array(batch[o_mid][Columns.OBS][OBSERVATIONS][o_s:o_s+o_l].cpu())
        assert np.abs(tmp-o_ep_obs).sum() == 0
        # Set the aug values
        lc = x_aug_self.shape[1]
        # First half of X's logits are X's aug.
        batch[x_mid][AGENT_LOGITS][x_s:x_s+x_l, :lc] = x_aug_self
        # Second half of X's aug are O's corresponding aug.
        batch[x_mid][AGENT_LOGITS][x_s:x_s+o_l, lc:] = o_aug_othr
        # First half of O's aug are O's aug
        batch[o_mid][AGENT_LOGITS][o_s:o_s+o_l, :lc] = o_aug_self
        # Second half of O's aug are X's aug downshifted by 1.
        batch[o_mid][AGENT_LOGITS][o_s:o_s+(x_l-1), lc:] = x_aug_othr[1:]

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: MultiRLModule,
        episodes: List[EpisodeType],
        batch: Dict[str, Any],
        **kwargs,
    ):
        # Device to place all GAE result tensors (advantages and value targets) on.
        device = None
        # Extract all single-agent episodes.
        sa_episodes_list = list(
            self.single_agent_episode_iterator(episodes, agents_that_stepped_only=False)
        )
        # Gather logits for next two actions, to inform critic about policies
        sc = rl_module[SHARED_CRITIC_ID]
        sa, oa = self.aug_fn_dict[sc.self_aug], self.aug_fn_dict[sc.other_aug]
        if (sa or oa):
          self.augment_critic(batch, episodes, sa, oa, sc.aug_size)
        # Perform the value net's forward pass.
        # Our only modification to the original - we use the shared critic to compute value, rather than the individual networks
        # When we add logits, we should also add other_agent_logits to each agent's associated batch.
        # Would probably suffice to get them from the main, un-processed logit buffers. The only observation that shouldn't have both is the terminating one, and it's easy to just mask it out based on terminations.
        # multi_agent_episode_id could serve as a fallback here.
        shared_critic = rl_module[SHARED_CRITIC_ID]
        vf_preds = rl_module.foreach_module(
            func=lambda mid, module: (
                # The critic class itself should handle the use of logits
                shared_critic.compute_values(batch[mid])
                if mid in batch and (mid != SHARED_CRITIC_ID) # not critic
                and isinstance(module, TorchRLModule) # trainable torch module
                else None
            ),
            return_dict=True,
        )
        # Loop through all modules and perform each one's GAE computation.
        for module_id, module_vf_preds in vf_preds.items():
            # Skip those outputs of RLModules that are not implementers of
            # `ValueFunctionAPI`.
            if module_vf_preds is None:
                continue

            module = rl_module[module_id]
            device = module_vf_preds.device
            # Convert to numpy for the upcoming GAE computations.
            module_vf_preds = convert_to_numpy(module_vf_preds)

            # Collect (single-agent) episode lengths for this particular module.
            episode_lens = [
                len(e) for e in sa_episodes_list if e.module_id in [None, module_id]
            ]

            # Remove all zero-padding again, if applicable, for the upcoming
            # GAE computations.
            # Unpadding doesn't matter here.
            module_vf_preds = unpad_data_if_necessary(episode_lens, module_vf_preds)
            # Compute value targets.
            module_value_targets = compute_value_targets(
                values=module_vf_preds,
                rewards=unpad_data_if_necessary(
                    episode_lens,
                    convert_to_numpy(batch[module_id][Columns.REWARDS]),
                ),
                terminateds=unpad_data_if_necessary(
                    episode_lens,
                    convert_to_numpy(batch[module_id][Columns.TERMINATEDS]),
                ),
                truncateds=unpad_data_if_necessary(
                    episode_lens,
                    convert_to_numpy(batch[module_id][Columns.TRUNCATEDS]),
                ),
                gamma=self.gamma,
                lambda_=self.lambda_,
            )
            assert module_value_targets.shape[0] == sum(episode_lens)

            module_advantages = module_value_targets - module_vf_preds
            # Drop vf-preds, not needed in loss. Note that in the DefaultPPORLModule,
            # vf-preds are recomputed with each `forward_train` call anyway to compute
            # the vf loss.
            # Standardize advantages (used for more stable and better weighted
            # policy gradient computations).
            module_advantages = (module_advantages - module_advantages.mean()) / max(
                1e-4, module_advantages.std()
            )

            # Zero-pad the new computations, if necessary.
            if module.is_stateful():
                module_advantages = np.stack(
                    split_and_zero_pad_n_episodes(
                        module_advantages,
                        episode_lens=episode_lens,
                        max_seq_len=module.model_config["max_seq_len"],
                    ),
                    axis=0,
                )
                module_value_targets = np.stack(
                    split_and_zero_pad_n_episodes(
                        module_value_targets,
                        episode_lens=episode_lens,
                        max_seq_len=module.model_config["max_seq_len"],
                    ),
                    axis=0,
                )
            batch[module_id][Postprocessing.ADVANTAGES] = module_advantages
            batch[module_id][Postprocessing.VALUE_TARGETS] = module_value_targets

        # Convert all GAE results to tensors.
        if self._numpy_to_tensor_connector is None:
            self._numpy_to_tensor_connector = NumpyToTensor(
                as_learner_connector=True, device=device
            )
        tensor_results = self._numpy_to_tensor_connector(
            rl_module=rl_module,
            batch={
                mid: {
                    Postprocessing.ADVANTAGES: module_batch[Postprocessing.ADVANTAGES],
                    Postprocessing.VALUE_TARGETS: (
                        module_batch[Postprocessing.VALUE_TARGETS]
                    ),
                }
                for mid, module_batch in batch.items()
                if vf_preds[mid] is not None
            },
            episodes=episodes,
        )
        # Move converted tensors back to `batch`.
        for mid, module_batch in tensor_results.items():
            batch[mid].update(module_batch)
        return batch