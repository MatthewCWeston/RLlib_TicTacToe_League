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
from algo.constants import SHARED_CRITIC_ID, AGENT_LOGITS, OBSERVATIONS, ACTION_MASK
from algo.modules.critic.SharedCriticCatalog import (
    LOGITS,
    ACTIONS,
    LOGITS_AND_ACTIONS,
)

# debug
from environments.TicTacToe import display_board
from collections import defaultdict

class CMAPPOGAEConnector(ConnectorV2):
    '''
        Convention for shared critic:
         - The 'viewpoint' player is main if the game is main v other, and X if the game is main v main.
            - We do this by checking if X's mid is 'main'. All checks for which is main should do this.
         - ID = 0 for main, 1 for exploiter, ID+2 for PFSP
         - How does interleaving work here? Do we need to deinterleave differently? (test with self-play)
         
         Other improvements:
          - In the self-play loop, we can set the weights for the Xth section equal to the weights of the X-1th section when a new agent is added.
    '''
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
            LOGITS_AND_ACTIONS: self.get_logits_and_actions,
            None: None
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

    def augment_critic(self, batch, meps, aug_fn_self, aug_fn_othr, aug_size, self_aug_size):
      for aid in batch:
        # Initialize next two action logit sets
        b_lgts = batch[aid][Columns.ACTION_DIST_INPUTS]
        batch[aid][AGENT_LOGITS] = torch.zeros((b_lgts.shape[0], aug_size)).to(b_lgts.device) # Create tensors to store logits
      start_indices = defaultdict(lambda: 0) # where to start in each agent's batch, when populating next action logits for critic
      lc = 0
      for mep in meps:
        x_ep, o_ep = mep.agent_episodes['X'], mep.agent_episodes['O']
        x_mid, o_mid = x_ep.module_id, o_ep.module_id
        x_l, o_l = len(x_ep), len(o_ep)
        # Start indices. We alternate because we might have the same module.
        x_s = start_indices[x_mid]
        start_indices[x_mid]+=x_l
        o_s = start_indices[o_mid]
        start_indices[o_mid]+=o_l
        # Grab aug vectors. 'self' and 'other' now refer to the main module and its opponent.
        use_next = 1 # Use the action of the next agent (ensures we'll always see something from the varied opponent)
        if (x_mid=='main'):
            if (aug_fn_self):
                aug = aug_fn_self(batch, x_mid, x_s, x_l)
                # First part of X's aug and O's aug come from X
                batch[x_mid][AGENT_LOGITS][x_s:x_s+x_l, :self_aug_size] = aug
                batch[o_mid][AGENT_LOGITS][o_s:o_s+(x_l-1), :self_aug_size] = aug[1:] * use_next
            if (aug_fn_othr):
                aug = aug_fn_othr(batch, o_mid, o_s, o_l)
                # Second parts of X's aug and O's aug come from O
                batch[x_mid][AGENT_LOGITS][x_s:x_s+o_l, self_aug_size:] = aug * use_next
                batch[o_mid][AGENT_LOGITS][o_s:o_s+o_l, self_aug_size:] = aug
        else:
            if (aug_fn_self):
                aug = aug_fn_self(batch, o_mid, o_s, o_l)
                # First part of X's aug and O's aug come from O
                batch[x_mid][AGENT_LOGITS][x_s:x_s+o_l, :self_aug_size] = aug * use_next
                batch[o_mid][AGENT_LOGITS][o_s:o_s+o_l, :self_aug_size] = aug
            if (aug_fn_othr):
                aug = aug_fn_othr(batch, x_mid, x_s, x_l)
                # Second parts of X's aug and O's aug come from X
                batch[x_mid][AGENT_LOGITS][x_s:x_s+x_l, self_aug_size:] = aug
                batch[o_mid][AGENT_LOGITS][o_s:o_s+(x_l-1), self_aug_size:] = aug[1:] * use_next

    def augment_critic_identity(self, batch, meps, aug_size):
      '''
          Provides the critic with a one-hot vector indicating the opponent's identity.
      '''
      for aid in batch:
        # Initialize next two action logit sets
        b_lgts = batch[aid][Columns.ACTION_DIST_INPUTS]
        batch[aid][AGENT_LOGITS] = torch.zeros((b_lgts.shape[0],aug_size), dtype=torch.long).to(b_lgts.device)
      start_indices = defaultdict(lambda: 0) # where to start in each agent's batch, when populating next action logits for critic
      lc = 0
      other_mids = defaultdict(lambda:0)
      for mep in meps:
        x_ep, o_ep = mep.agent_episodes['X'], mep.agent_episodes['O']
        x_mid, o_mid = x_ep.module_id, o_ep.module_id
        x_l, o_l = len(x_ep), len(o_ep)
        # Start indices. We alternate because we might have the same module.
        x_s = start_indices[x_mid]
        start_indices[x_mid]+=x_l
        o_s = start_indices[o_mid]
        start_indices[o_mid]+=o_l
        other_mid = (o_mid if x_mid=='main' else x_mid)
        if ('main_v' in other_mid):
            other_mid = int(other_mid.split('main_v')[-1]) + 2
        else:
            other_mid = 0 if other_mid=='main' else 1
        other_mids[other_mid] += 1
        other_mid = torch.nn.functional.one_hot(torch.tensor(other_mid), aug_size)
        # First part of X's aug and O's aug come from X
        batch[x_mid][AGENT_LOGITS][x_s:x_s+x_l] = other_mid
        batch[o_mid][AGENT_LOGITS][o_s:o_s+o_l] = other_mid
      print("Other module appearances this epoch: ")
      print(other_mids)

    def call_with_interleaving(
        self,
        rl_module: MultiRLModule,
        episodes: List[EpisodeType],
        batch: Dict[str, Any],
        **kwargs,
    ):
        sc = rl_module[SHARED_CRITIC_ID]
        # Get total length
        tl = 0
        obs_size = 0
        device = batch['main'][ACTION_MASK].device
        am_size = batch['main'][ACTION_MASK].shape[1]
        for mid, b in batch.items():
          if (Columns.OBS in b):
            obs_shp = b[Columns.OBS].shape
            obs_size = obs_shp[1]
            tl += obs_shp[0]
        # Create the critic's own batch
        critic_batch = {Columns.OBS: torch.zeros((tl, obs_size)), ACTION_MASK: torch.zeros((tl, am_size)), Columns.REWARDS: torch.zeros((tl)), Columns.TERMINATEDS: torch.full((tl,), False), Columns.TRUNCATEDS: torch.full((tl,), False)}
        cb_s = 0 # start index for critic_batch
        # for splitting the list later
        module_ixs = defaultdict(lambda: [])
        start_indices = defaultdict(lambda: 0) # where to start in each batch
        multipliers = np.ones((tl,)) # 
        cols_to_interleave = [Columns.OBS, Columns.REWARDS, Columns.TERMINATEDS, Columns.TRUNCATEDS]
        # Augment observations if requested
        if (sc.identity_aug):
          self.augment_critic_identity(batch, episodes, sc.aug_size)
          critic_batch[AGENT_LOGITS] = torch.zeros((tl, batch['main'][AGENT_LOGITS].shape[1]), dtype=torch.long)
        else:
          sa, oa = self.aug_fn_dict[sc.self_aug], self.aug_fn_dict[sc.other_aug]
          self.augment_critic(batch, episodes, sa, oa, sc.aug_size, sc.self_aug_size)
          critic_batch[AGENT_LOGITS] = torch.zeros((tl, batch['main'][AGENT_LOGITS].shape[1]))
        if (AGENT_LOGITS in batch['main']):
          cols_to_interleave.append(AGENT_LOGITS)
        for k, v in critic_batch.items(): # critic batch to device
            critic_batch[k] = v.to(device)
        for mep in episodes: # populate critic batch
          x_ep, o_ep = mep.agent_episodes['X'], mep.agent_episodes['O']
          x_mid, o_mid = x_ep.module_id, o_ep.module_id
          # We need to handle main v main. X goes first, so update X's pointer before accessing O.
          x_l, o_l = len(x_ep), len(o_ep)
          x_s = start_indices[x_mid]
          start_indices[x_mid]+=x_l
          o_s = start_indices[o_mid]
          start_indices[o_mid]+=o_l
          # Set indices (in critic_batch) for each episode to be sent into
          # x gets start index and then counts up by 2, o gets 1 and then same
          x_ixs = torch.arange(cb_s, cb_s+x_l+o_l, 2)
          o_ixs = torch.arange(cb_s+1, cb_s+x_l+o_l, 2)
          cb_s += x_l + o_l
          # Set main/other eps
          if (x_mid=='main'):
            main_ep, other_ep = x_ep, o_ep
            m_ixs, o_ixs = x_ixs, o_ixs
            m_s, m_l, o_s, o_l = x_s, x_l, o_s, o_l
          else:
            main_ep, other_ep = o_ep, x_ep
            m_ixs, o_ixs = o_ixs, x_ixs
            m_s, m_l, o_s, o_l = o_s, o_l, x_s, x_l
            o_mid = x_mid
          mb, ob = batch['main'], batch[o_mid]
          # Track indices for sending back advantages and VTs
          module_ixs['main'].extend(m_ixs)
          module_ixs[o_mid].extend(o_ixs)
          # Get data, populate critic_batch
          for c in cols_to_interleave:
            m_data, o_data = mb[c][m_s:m_s+m_l], ob[c][o_s:o_s+o_l]
            # Handle special cases (maybe go by main/other rather than x/o)
            if (c == Columns.OBS):
              # Invert observation of non-main agent
              o_data = torch.cat((o_data[:,9:], o_data[:,:9]), dim=1)
            elif (c == Columns.REWARDS): # Invert opponent rewards
              o_data *= -1
            if (c not in [Columns.OBS, AGENT_LOGITS]):
              # Ignore termination flags that aren't the last one, and the reward given to the agent that didn't move last
              if (m_ixs[-1] > o_ixs[-1]):
                critic_batch[c][m_ixs] = m_data
              else:
                critic_batch[c][o_ixs] = o_data
            else:
              # Put data into critic batch
              critic_batch[c][m_ixs], critic_batch[c][o_ixs] = m_data, o_data
              multipliers[o_ixs] = -1 # multiply values at opponent indices by negative 1.
        # Calculate value targets
        with torch.no_grad():
          vf_preds = sc.compute_values(critic_batch)
        #
        vf_preds = convert_to_numpy(vf_preds)
        value_targets = compute_value_targets(
            values=vf_preds,
            rewards=convert_to_numpy(critic_batch[Columns.REWARDS]),
            terminateds=convert_to_numpy(critic_batch[Columns.TERMINATEDS]),
            truncateds=convert_to_numpy(critic_batch[Columns.TRUNCATEDS]),
            gamma=self.gamma,
            lambda_=self.lambda_,
        )
        advantages = value_targets - vf_preds
        advantages = (advantages - advantages.mean()) / max(
            1e-4, advantages.std()
        )
        critic_batch[Postprocessing.VALUE_TARGETS] = value_targets
        critic_batch[Postprocessing.ADVANTAGES] = advantages # temp
        # Negate value targets (not used) and advantages for actors
        actor_value_targets = value_targets * multipliers
        actor_advantages = advantages * multipliers
        # Get VTs for each module. Remember to reverse VTs for non-main module
        #for mid, ixs, mult in zip(['main', o_mid], [m_ixs, o_ixs], [1,-1]):
        for mid in batch.keys():
          module = rl_module[mid]
          if (mid == SHARED_CRITIC_ID) or not isinstance(module, TorchRLModule):
            continue
          # multiplier is 1 if mid is main and (both are not main OR batch is X).
          ixs = module_ixs[mid]
          module_advantages= actor_advantages[ixs]
          module_vts= actor_value_targets[ixs]
          '''mult = 1 if mid == 'main' else -1 # Negate other modules' VTs
          module_advantages= advantages[ixs]*mult
          module_vts= value_targets[ixs]*mult'''
          # Set module VTs and advantages
          batch[mid][Postprocessing.ADVANTAGES] = module_advantages
          batch[mid][Postprocessing.VALUE_TARGETS] = module_vts
        # Add critic batch
        batch[SHARED_CRITIC_ID] = critic_batch
        '''# Display code
        for i in range(20):
          print(f'R: {critic_batch[Columns.REWARDS][i]} VT: {critic_batch[Postprocessing.VALUE_TARGETS][i]}')
          print(critic_batch[Columns.TERMINATEDS][i])
          display_board(critic_batch[Columns.OBS][i])
        # Display code
        print("MAIN")
        for i in range(20):
          print(f'R: {batch["main"][Columns.REWARDS][i]} VT: {batch["main"][Postprocessing.VALUE_TARGETS][i]}')
          print(batch['main'][Columns.TERMINATEDS][i])
          display_board(batch['main'][Columns.OBS][i])
        raise Exception() # '''
        return device

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: MultiRLModule,
        episodes: List[EpisodeType],
        batch: Dict[str, Any],
        **kwargs,
    ):
        # Reformat to purge action masks (since we run value computation on the critic's batch, this doesn't happen automatically for other modules)
        for mid, module_batch in batch.items():
          if (Columns.OBS in module_batch):
            cobs = module_batch[Columns.OBS]
            if (ACTION_MASK in cobs): # Action-masking
              module_batch[Columns.OBS] = cobs[OBSERVATIONS]
              module_batch[ACTION_MASK] = cobs[ACTION_MASK]
        # Compute module value targets and advantages with interleaving
        device = self.call_with_interleaving(rl_module,episodes,batch,**kwargs)
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
                if (Postprocessing.ADVANTAGES in batch[mid])
            },
            episodes=episodes,
        )
        # Move converted tensors back to `batch`.
        for mid, module_batch in tensor_results.items():
            batch[mid].update(module_batch)
        return batch