import torch
from torch.utils.data import Dataset

class OfflineWorkspaceDataset(Dataset):
    def __init__(self, size=100, img_shape=(3, 224, 224)):
        self.size = size
        self.img_shape = img_shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.rand(*self.img_shape)
        instruction = "manipulate the object"
        target_action = torch.randint(0, 7, ()).long()
        return x, instruction, target_action