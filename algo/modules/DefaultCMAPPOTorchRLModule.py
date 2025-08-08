# @title DefaultCMAPPOTorchRLModule
from typing import Any, Dict, Optional

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
from algo.modules.CMAPPOCatalog import CMAPPOCatalog
from algo.modules.DefaultCMAPPORLModule import DefaultCMAPPORLModule
from algo.modules.critic.SharedCriticCatalog import SharedCriticCatalog


@DeveloperAPI
class DefaultCMAPPOTorchRLModule(TorchRLModule, DefaultCMAPPORLModule):
    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = CMAPPOCatalog
        super().__init__(*args, **kwargs, catalog_class=catalog_class)

    @override(RLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Default forward pass (used for inference and exploration)."""
        output = {}
        # Encoder forward pass.
        encoder_outs = self.encoder(batch)
        # Stateful encoder?
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]
        # Pi head.
        output[Columns.ACTION_DIST_INPUTS] = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        return output

    @override(RLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Train forward pass (keep embeddings for possible shared value func. call)."""
        output = {}
        encoder_outs = self.encoder(batch)
        output[Columns.EMBEDDINGS] = encoder_outs[ENCODER_OUT][CRITIC]
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]
        output[Columns.ACTION_DIST_INPUTS] = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        return output