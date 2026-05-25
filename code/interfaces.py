from abc import ABC, abstractmethod
from typing import Any, Dict

class IMinimaxStrategy(ABC):
    @abstractmethod
    def approximate_worst_case_noise(self, base_loss: Any, patch_parameters: Any) -> Any: pass
    @abstractmethod
    def apply_perturbation(self, parameters: Any, noise: Any) -> Any: pass

class IKinematicSimulator(ABC):
    @abstractmethod
    def sample_physical_transform(self) -> Any: pass
    @abstractmethod
    def construct_augmented_observation(self, clean_frame: Any, patch: Any, transform: Any) -> Any: pass

class IPhysicalConstraint(ABC):
    @abstractmethod
    def compute_penalty(self, patch_parameters: Any) -> Any: pass

class IVLAWhiteBox(ABC):
    @abstractmethod
    def predict_kinematics(self, observation: Any, instruction: str) -> Any: pass
    @abstractmethod
    def compute_kinematic_loss(self, predicted_action: Any, target_action: Any) -> Any: pass