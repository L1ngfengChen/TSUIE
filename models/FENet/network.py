import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from models.FENet.loss import PerceptualLoss, SSIMLoss


class NetWork(nn.Module):
    def __init__(self, Net, device):
        super().__init__()
        self.device = torch.device(device)
        self.Net = Net.to(self.device)

        self.L1Loss = nn.L1Loss().to(self.device)
        self.PerceptionLoss = PerceptualLoss().to(self.device)
        self.SSIMLoss = SSIMLoss().to(self.device)

    def forward(self, x, y, label = None, Training = True):
        if Training:
            x1 = self.Net(x,y)
            loss = self.L1Loss(x1, label)+0.01*self.PerceptionLoss(x1, label)+(1-self.SSIMLoss(x1, label))
            return loss
        
        else:
            return self.Net(x, y)
