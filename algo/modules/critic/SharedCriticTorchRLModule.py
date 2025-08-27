import typing
from typing import Any, Optional

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
from algo.constants import AGENT_LOGITS
from algo.modules.critic.SharedCriticRLModule import SharedCriticRLModule
from algo.modules.critic.SharedCriticCatalog import SharedCriticCatalog #, ID_EMBEDDING_SIZE

@DeveloperAPI
class SharedCriticTorchRLModule(TorchRLModule, SharedCriticRLModule):
    def __init__(self, *args, **kwargs):
        catalog_class = kwargs.pop("catalog_class", None)
        if catalog_class is None:
            catalog_class = SharedCriticCatalog
        super().__init__(*args, **kwargs, catalog_class=catalog_class)

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: typing.Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        if (self.identity_aug):
            #identity_emb = self.identity_emb(batch[AGENT_LOGITS])
            identity_emb = batch[AGENT_LOGITS]
            batch = {Columns.OBS: torch.cat((batch[Columns.OBS], identity_emb), dim=1)}
        elif (self.self_aug or self.other_aug): # Replace batch with augmented batch
            batch = {Columns.OBS: torch.cat((batch[Columns.OBS], batch[AGENT_LOGITS]), dim=1)}
        if embeddings is None:
            embeddings = self.encoder(batch)[ENCODER_OUT][CRITIC]
        # Value head.
        vf_out = self.vf(embeddings)
        # Debugging for propagation of embeddings
        '''if (vf_out.shape[0] > 128 and self.identity_aug):
            print('='*20)
            print("COMPUTE VALUES CALLED")
            print(self.encoder.encoder.net.mlp[0].weight.abs().sum(dim=0)[-self.aug_size:])'''
        # Squeeze out last dimension (single node value head).
        return vf_out.squeeze(-1)
    
    def new_agent_embedding(self, ix):
        ''' Set embedding at index ix to embedding at index ix-1 (a new frozen policy was initialized) '''
        with torch.no_grad():
            first_layer = self.encoder.encoder.net.mlp[0].weight
            first_layer[:, ix] = first_layer[:, ix-1].clone()