import torch
from torch.utils.data import DataLoader
from dataset import OfflineWorkspaceDataset
from models import FrozenVLAModel
from psam import PSAM

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = OfflineWorkspaceDataset(size=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = FrozenVLAModel().to(device)
    model.eval()
    
    psam_optimizer = PSAM(model, alpha=0.01, rho=0.05, lambda_1=0.1, lambda_2=0.1)
    
    delta = torch.rand(1, 3, 224, 224, device=device)
    base_mask = torch.zeros(1, 1, 224, 224, device=device)
    base_mask[:, :, 100:124, 100:124] = 1.0
    
    epochs = 10
    for epoch in range(epochs):
        for x, instructions, a_target in dataloader:
            x = x.to(device)
            a_target = a_target.to(device)
            delta = psam_optimizer.step(delta, x, base_mask, instructions, a_target)

if __name__ == "__main__":
    main()