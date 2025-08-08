# @title SharedCriticCatalog (added logit handling)
# __sphinx_doc_begin__
import gymnasium as gym

from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import (
    ActorCriticEncoderConfig,
    MLPHeadConfig,
    FreeLogStdMLPHeadConfig,
)
from ray.rllib.core.models.base import Encoder, ActorCriticEncoder, Model
from ray.rllib.utils import override
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic

from ray.rllib.algorithms.ppo.ppo_catalog import _check_if_diag_gaussian

class SharedCriticCatalog(Catalog):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space, # TODO: Remove?
        model_config_dict: dict,
    ):
        """Initializes the PPOCatalog.

        Args:
            observation_space: The observation space of the Encoder.
            action_space: The action space for the Pi Head.
            model_config_dict: The model config to use.
        """
        super().__init__(
            observation_space=observation_space,
            action_space=action_space, # We shouldn't need to provide this. I should reconfigure at some point.
            model_config_dict=model_config_dict,
        )
        # Are we expecting action logits?
        self.use_logits = self._model_config_dict["use_logits"] if "use_logits" in self._model_config_dict else False
        self.use_actions = self._model_config_dict["use_actions"] if "use_actions" in self._model_config_dict else False
        logits_multiplier = int(self.use_logits) + int(self.use_actions)
        if (logits_multiplier):
            self._encoder_config.input_dims = (self._model_config_dict["logits_size"]*logits_multiplier + self._encoder_config.input_dims[0],)
        # We only want one encoder, so don't make a separate one for an actor that doesn't exist. We should switch to a base_encoder_config at some point.
        self._encoder_config.shared = True
        # Replace EncoderConfig by ActorCriticEncoderConfig
        self.encoder_config = ActorCriticEncoderConfig(
            base_encoder_config=self._encoder_config,
            shared=True, # Because we only want one network
        )
        self.vf_head_hiddens = self._model_config_dict["head_fcnet_hiddens"]
        self.vf_head_activation = self._model_config_dict[
            "head_fcnet_activation"
        ]

        self.vf_head_config = MLPHeadConfig(
            input_dims=self.latent_dims,
            hidden_layer_dims=self.vf_head_hiddens,
            hidden_layer_activation=self.vf_head_activation,
            output_layer_activation="linear",
            output_layer_dim=1,
        )

    @override(Catalog)
    def build_encoder(self, framework: str) -> Encoder:
        """Builds the encoder.

        Since PPO uses an ActorCriticEncoder, this method should not be implemented.
        """
        return self.encoder_config.build(framework=framework)

    @OverrideToImplementCustomLogic
    def build_vf_head(self, framework: str) -> Model:
        """Builds the value function head.

        The default behavior is to build the head from the vf_head_config.
        This can be overridden to build a custom value function head as a means of
        configuring the behavior of a PPORLModule implementation.

        Args:
            framework: The framework to use. Either "torch" or "tf2".

        Returns:
            The value function head.
        """
        return self.vf_head_config.build(framework=framework)


# __sphinx_doc_end__