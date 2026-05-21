import torch

def tv_loss(delta):
    diff_i = delta[:, :, 1:, :-1] - delta[:, :, :-1, :-1]
    diff_j = delta[:, :, :-1, 1:] - delta[:, :, :-1, :-1]
    return torch.sum(torch.sqrt(diff_i ** 2 + diff_j ** 2 + 1e-8))

def nps_loss(delta, cmyk_colors):
    cmyk_colors = cmyk_colors.to(delta.device)
    b, c, h, w = delta.shape
    delta_flat = delta.view(b, c, -1).permute(0, 2, 1).unsqueeze(-2)
    cmyk_colors_flat = cmyk_colors.view(1, 1, -1, 3)
    distances = torch.norm(delta_flat - cmyk_colors_flat, dim=-1)
    min_distances, _ = torch.min(distances, dim=-1)
    return torch.sum(min_distances)