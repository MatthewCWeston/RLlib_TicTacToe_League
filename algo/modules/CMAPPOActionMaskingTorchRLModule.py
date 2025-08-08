import gymnasium as gym
from typing import Dict, Optional, Tuple, Union

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingRLModule,
)

torch, nn = try_import_torch()

# our code
from algo.modules.DefaultCMAPPOTorchRLModule import DefaultCMAPPOTorchRLModule
from algo.modules.CMAPPOCatalog import CMAPPOCatalog
from algo.constants import ACTION_MASK, OBSERVATIONS

class CMAPPOActionMaskingTorchRLModule(ActionMaskingRLModule, DefaultCMAPPOTorchRLModule):
    @override(DefaultCMAPPOTorchRLModule)
    def setup(self):
        super().setup()
        # We need to reset here the observation space such that the
        # super`s (`PPOTorchRLModule`) observation space is the
        # original space (i.e. without the action mask) and `self`'s
        # observation space contains the action mask.
        self.observation_space = self.observation_space_with_mask

    @override(DefaultCMAPPOTorchRLModule)
    def _forward_inference(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        # Preprocess the original batch to extract the action mask.
        action_mask, batch = self._preprocess_batch(batch)
        # Run the forward pass.
        outs = super()._forward_inference(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, action_mask)

    @override(DefaultCMAPPOTorchRLModule)
    def _forward_exploration(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        # Preprocess the original batch to extract the action mask.
        action_mask, batch = self._preprocess_batch(batch)
        # Run the forward pass.
        outs = super()._forward_exploration(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, action_mask)

    @override(DefaultCMAPPOTorchRLModule)
    def _forward_train(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        # Run the forward pass.
        outs = super()._forward_train(batch, **kwargs)
        # Mask the action logits and return.
        return self._mask_action_logits(outs, batch[ACTION_MASK])

    def _preprocess_batch(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Tuple[TensorType, Dict[str, TensorType]]:
        # Extract the available actions tensor from the observation.
        action_mask = batch[Columns.OBS].pop(ACTION_MASK)

        # Modify the batch for the `DefaultPPORLModule`'s `forward` method, i.e.
        # pass only `"obs"` into the `forward` method.
        batch[Columns.OBS] = batch[Columns.OBS].pop(OBSERVATIONS)

        # Return the extracted action mask and the modified batch.
        return action_mask, batch

    def _mask_action_logits(
        self, batch: Dict[str, TensorType], action_mask: TensorType
    ) -> Dict[str, TensorType]:
        # Convert action mask into an `[0.0][-inf]`-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)

        # Mask the logits.
        batch[Columns.ACTION_DIST_INPUTS] += inf_mask

        # Return the batch with the masked action logits.
        return batch