import torch
import torch.nn as nn

def zero_weights_fn(m):
    if type(m) == nn.Linear:
        m.weight.data.fill_(0)
        m.bias.data.fill_(0)

class Resblock(nn.Module):
    def __init__(self, in_features, middle_features, out_features, downsample=False, include_residual=True, zero_weights=False):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_features, middle_features, kernel_size=(1,1),
                                        stride=2 if downsample else 1, bias=False),
            nn.BatchNorm2d(middle_features),
            nn.ReLU(),
            nn.Conv2d(middle_features, middle_features, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(middle_features),
            nn.ReLU(),
            nn.Conv2d(middle_features, out_features, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )
        if zero_weights:
            self.network.apply(zero_weights_fn)
        self.project = False
        self.downsample = False
        if in_features != out_features:
            self.projector = nn.Conv2d(in_features, out_features, kernel_size=(1,1), bias=False)
            self.project = True
        if downsample:
            self.downsampler = nn.MaxPool2d(2,2)
            self.downsample = True
        self.include_residual = include_residual
    def forward(self, xb):
        block_out = self.network(xb)
        if not self.include_residual:
            return block_out
        if self.project:
            xb = self.projector(xb)
        if self.downsample:
            xb = self.downsampler(xb)
        return block_out + xb

class ResNet50(nn.Module):
    def __init__(self, include_residual=True, zero_weights=False, num_classes=100):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(3,64, kernel_size=(3,3), padding=1))
        num_residblocks = [3,4,6,3]
        block_info = [[64,256],[128,512],[256,1024],[512,2048]]
        latest_out_featcount = 64
        for i in range(4): ##hard coded 4 because its the # of groups of different resblock types
            for j in range(num_residblocks[i]):
                self.network.append(Resblock(latest_out_featcount,*block_info[i],
                                        downsample= True if (j==0 and i > 0) else False, include_residual=include_residual, zero_weights=zero_weights))
                latest_out_featcount = block_info[i][-1]
        self.network.append(nn.AdaptiveAvgPool2d(1))
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )
    def forward(self, xb):
        return self.fc(self.network(xb))