from typing import Dict

from ray.rllib.algorithms.ppo.default_ppo_rl_module import DefaultPPORLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
from ray.util.annotations import DeveloperAPI

torch, nn = try_import_torch()

# our code
from algo.modules.critic.SharedCriticTorchRLModule import SharedCriticTorchRLModule
from algo.modules.critic.SharedCriticCatalog import SharedCriticCatalog
from algo.constants import ACTION_MASK, OBSERVATIONS

@DeveloperAPI
class ActionMaskingSharedCriticTorchRLModule(SharedCriticTorchRLModule):
    def __init__(self, *args, **kwargs):
        try:
          print("critic init start")
          catalog_class = kwargs.pop("catalog_class", None)
          observation_space = kwargs.pop("observation_space", None)
          if catalog_class is None:
              catalog_class = SharedCriticCatalog
          self.observation_space_with_mask = observation_space
          self.observation_space = observation_space[OBSERVATIONS]
          super().__init__(*args, **kwargs, catalog_class=catalog_class, observation_space=self.observation_space)
        except Exception as e:
          print(e)

    @override(SharedCriticTorchRLModule)
    def setup(self):
        super().setup()
        self.observation_space = self.observation_space_with_mask

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, TensorType], embeddings=None):
        # Check, if the observations are still in `dict` form.
        if isinstance(batch[Columns.OBS], dict):
            # Preprocess the batch to extract the `observations` to `Columns.OBS`.
            action_mask = batch[Columns.OBS].pop(ACTION_MASK)
            batch[Columns.OBS] = batch[Columns.OBS].pop(OBSERVATIONS)
            # NOTE: Because we manipulate the batch we need to add the `action_mask`
            # to the batch to access them in `_forward_train`.
            batch[ACTION_MASK] = action_mask
        # Call the super's method to compute values for GAE.
        return super().compute_values(batch, embeddings)

