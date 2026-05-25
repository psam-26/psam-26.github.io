from abc import abstractmethod
from typing import Any
from src.psam.core.interfaces import IVLAWhiteBox

class IVisionEncoder(ABC):
    @abstractmethod
    def extract_spatial_features(self, pixel_tensor: Any) -> Any:
        pass

class ILanguageProjector(ABC):
    @abstractmethod
    def embed_instruction(self, text: str) -> Any:
        pass
    
    @abstractmethod
    def cross_attend(self, visual_features: Any, textual_features: Any) -> Any:
        pass

class IActionDecoder(ABC):
    @abstractmethod
    def decode_continuous_kinematics(self, fused_embeddings: Any) -> Any:
        pass

class AbstractFrozenVLA(IVLAWhiteBox):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.vision_backbone = self._initialize_vision_encoder()
        self.language_backbone = self._initialize_language_projector()
        self.action_head = self._initialize_action_decoder()
        self._freeze_all_parameters()
        self._configure_gradient_checkpointing()

    def predict_kinematics(self, observation: Any, instruction: str) -> Any:
        visual_features = self.vision_backbone.extract_spatial_features(observation)
        textual_features = self.language_backbone.embed_instruction(instruction)
        fused_representation = self.language_backbone.cross_attend(visual_features, textual_features)
        action_logits = self.action_head.decode_continuous_kinematics(fused_representation)
        return self._map_logits_to_continuous_7dof(action_logits)

    def compute_kinematic_loss(self, predicted_action: Any, target_action: Any) -> Any:
        deviation_matrix = self._calculate_trajectory_deviation(predicted_action, target_action)
        normalized_loss = self._apply_dof_normalization(deviation_matrix)
        return self._aggregate_loss_components(normalized_loss)

    @abstractmethod
    def _initialize_vision_encoder(self) -> IVisionEncoder:
        pass

    @abstractmethod
    def _initialize_language_projector(self) -> ILanguageProjector:
        pass

    @abstractmethod
    def _initialize_action_decoder(self) -> IActionDecoder:
        pass

    @abstractmethod
    def _freeze_all_parameters(self) -> None:
        pass

    @abstractmethod
    def _configure_gradient_checkpointing(self) -> None:
        pass

    @abstractmethod
    def _map_logits_to_continuous_7dof(self, logits: Any) -> Any:
        pass

    @abstractmethod
    def _calculate_trajectory_deviation(self, pred: Any, target: Any) -> Any:
        pass

    @abstractmethod
    def _apply_dof_normalization(self, matrix: Any) -> Any:
        pass

    @abstractmethod
    def _aggregate_loss_components(self, norm_loss: Any) -> Any:
        pass