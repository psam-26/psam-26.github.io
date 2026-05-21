import torch
import torch.nn.functional as F
import random
import math

class PhysicalTransform:
    def __init__(self, max_angle=15, max_scale=0.2, max_translate=0.1):
        self.max_angle = max_angle
        self.max_scale = max_scale
        self.max_translate = max_translate

    def get_params(self, b):
        params = []
        for _ in range(b):
            angle = random.uniform(-self.max_angle, self.max_angle)
            scale = random.uniform(1 - self.max_scale, 1 + self.max_scale)
            tx = random.uniform(-self.max_translate, self.max_translate)
            ty = random.uniform(-self.max_translate, self.max_translate)
            brightness = random.uniform(0.8, 1.2)
            params.append({
                "angle": angle, "scale": scale, "tx": tx, "ty": ty,
                "brightness": brightness
            })
        return params

    def apply(self, patch, mask, params):
        b, _, h, w = patch.shape
        t_patches = []
        t_masks = []
        device = patch.device
        
        for i in range(b):
            p = patch[i:i+1]
            m = mask[i:i+1]
            param = params[i]
            
            angle_rad = param["angle"] * math.pi / 180.0
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            tx = param["tx"]
            ty = param["ty"]
            scale = param["scale"]
            
            theta = torch.tensor([
                [cos_a / scale, -sin_a / scale, tx],
                [sin_a / scale,  cos_a / scale, ty]
            ], dtype=torch.float32, device=device).unsqueeze(0)
            
            grid = F.affine_grid(theta, p.size(), align_corners=False)
            
            p_t = F.grid_sample(p, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            m_t = F.grid_sample(m, grid, mode='nearest', padding_mode='zeros', align_corners=False)
            
            p_t = p_t * param["brightness"]
            p_t = torch.clamp(p_t, 0.0, 1.0)
            
            t_patches.append(p_t.squeeze(0))
            t_masks.append(m_t.squeeze(0))
            
        return torch.stack(t_patches), torch.stack(t_masks)