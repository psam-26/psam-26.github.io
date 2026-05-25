from src.psam.core.interfaces import IPhysicalConstraint
from typing import Any

class TotalVariationConstraint(IPhysicalConstraint):
    def __init__(self, lambda_weight: float):
        self.weight = lambda_weight

    def compute_penalty(self, patch_parameters: Any) -> Any:
        diff_horizontal, diff_vertical = self._compute_spatial_gradients(patch_parameters)
        
        tv_magnitude = self._calculate_euclidean_magnitude(diff_horizontal, diff_vertical)
        
        return self._scale_tensor(self._sum_all(tv_magnitude), self.weight)
        
    def _compute_spatial_gradients(self, tensor: Any) -> Any: pass
    def _calculate_euclidean_magnitude(self, dx: Any, dy: Any) -> Any: pass
    def _sum_all(self, tensor: Any) -> Any: pass
    def _scale_tensor(self, tensor: Any, weight: float) -> Any: pass


class NonPrintabilityScoreConstraint(IPhysicalConstraint):
    def __init__(self, lambda_weight: float, gamut_profile: Any):
        self.weight = lambda_weight
        self.gamut = gamut_profile

    def compute_penalty(self, patch_parameters: Any) -> Any:
        pairwise_distances = self._compute_pairwise_cdist(patch_parameters, self.gamut)
        
        min_distances = self._reduce_min(pairwise_distances)
        
        return self._scale_tensor(self._sum_all(min_distances), self.weight)

    def _compute_pairwise_cdist(self, pixels: Any, gamut: Any) -> Any: pass
    def _reduce_min(self, distances: Any) -> Any: pass
    def _sum_all(self, tensor: Any) -> Any: pass
    def _scale_tensor(self, tensor: Any, weight: float) -> Any: pass