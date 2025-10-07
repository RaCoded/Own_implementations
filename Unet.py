import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
                                  nn.BatchNorm2d(out_channels), # Permet d'ajouter de la stabilité lors de l'entrainement
                                  nn.ReLU(inplace=True)) #Introduction de non linéarité, inplace réduit l'utilisation mémoire, à prévilégier
    def forward(self,x):
        return self.conv(x)
    
# Module qui permet d'effectuer une double convolution normalisée avec introduction de non-linéarité
class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.doubleConv = nn.Sequential(Conv(in_channels,out_channels),
                                        Conv(out_channels,out_channels))
    def forward(self,x):
        return self.doubleConv(x)

# Module permettant la descente de l'encodeur
class Down(nn.Module()):
    def __init__(self,in_channels,out_channels):
        super().__init()
        self.down = nn.Sequential(nn.MaxPool2d(kernel_size=2), # Rédution par deux au niveau spatial
                                  DoubleConv(in_channels,out_channels))
    def forward(self,x):
        return self.down(x)
    
    
class Up(nn.Module()):
    