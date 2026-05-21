import torch
import torch.nn.functional as F
from transforms import PhysicalTransform
from losses import tv_loss, nps_loss

class PSAM:
    def __init__(self, model, alpha=0.01, rho=0.05, lambda_1=0.1, lambda_2=0.1):
        self.model = model
        self.alpha = alpha
        self.rho = rho
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.cmyk_colors = torch.tensor([
            [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]
        ])
        self.transform = PhysicalTransform()

    def step(self, delta, x, base_mask, instructions, a_target):
        delta.requires_grad_(True)
        b = x.shape[0]
        
        delta_expanded = delta.expand(b, -1, -1, -1)
        mask_expanded = base_mask.expand(b, -1, -1, -1)
        
        params = self.transform.get_params(b)
        t_delta, m_tau = self.transform.apply(delta_expanded, mask_expanded, params)
        
        o = (1 - m_tau) * x + m_tau * t_delta
        
        pred_base = self.model(o, instructions)
        l_base = F.cross_entropy(pred_base, a_target)
        
        grad_delta = torch.autograd.grad(l_base, delta, retain_graph=True)[0]
        epsilon_star = self.rho * grad_delta / (torch.norm(grad_delta, p=2) + 1e-8)
        epsilon_star = epsilon_star.detach()
        
        delta_hat = delta + epsilon_star
        delta_hat_expanded = delta_hat.expand(b, -1, -1, -1)
        
        t_delta_hat, m_tau_noisy = self.transform.apply(delta_hat_expanded, mask_expanded, params)
        
        o_noisy = (1 - m_tau_noisy) * x + m_tau_noisy * t_delta_hat
        
        pred_noisy = self.model(o_noisy, instructions)
        l_base_noisy = F.cross_entropy(pred_noisy, a_target)
        
        l_tv = tv_loss(delta)
        l_nps = nps_loss(delta, self.cmyk_colors)
        
        loss_final = l_base_noisy + self.lambda_1 * l_tv + self.lambda_2 * l_nps
        
        final_grad = torch.autograd.grad(loss_final, delta)[0]
        
        with torch.no_grad():
            delta_new = delta - self.alpha * final_grad
            delta_new = torch.clamp(delta_new, 0.0, 1.0)
            
        return delta_new.detach()