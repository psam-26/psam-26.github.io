from src.psam.core.interfaces import IMinimaxStrategy
from typing import Any

class TaylorExpansionMinimax(IMinimaxStrategy):
    def __init__(self, rho_radius: float):
        self.rho = rho_radius

    def approximate_worst_case_noise(self, base_loss: Any, patch_parameters: Any) -> Any:
        gradients = self._compute_gradients(base_loss, patch_parameters)
        
        grad_norm = self._compute_l2_norm(gradients)
        
        worst_case_noise = self._scale_and_normalize(gradients, grad_norm, self.rho)
        
        return worst_case_noise

    def apply_perturbation(self, parameters: Any, noise: Any) -> Any:
        return self._add_tensors(parameters, noise)

    # --- Abstract Mathematical Operations ---
    def _compute_gradients(self, loss: Any, parameters: Any) -> Any: pass
    def _compute_l2_norm(self, tensor: Any) -> Any: pass
    def _scale_and_normalize(self, tensor: Any, norm: Any, scale: float) -> Any: pass
    def _add_tensors(self, t1: Any, t2: Any) -> Any: pass