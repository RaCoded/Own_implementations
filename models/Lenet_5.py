import torch
import torch.nn as nn

class Lenet(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        # Definition de la couche c1
        self.c1 = nn.Conv2d(1,6,kernel_size=5)
        self.c2 = nn.MaxPool2d(2,stride=2)
        self.c3 = nn.Conv2d(6,16,kernel_size=5)
        self.c4 = nn.MaxPool2d(2,stride=2)
        # pas sur?
        self.c5 = nn.Conv2d(16,120,kernel_size=5)
        self.c6 = nn.Linear(120,84)
        self.c7 = nn.Linear(84,num_classes)
        
    def forward(self,x): #Possibilité d'ajouter du tanh, relu ou gelu !
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)