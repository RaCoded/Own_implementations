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
class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(kernel_size=2), # Rédution par deux au niveau spatial
                                  DoubleConv(in_channels,out_channels))
    def forward(self,x):
        return self.down(x)
    
# Module permettant la montée du décodeur
class Up(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2) # permet de réduire le nombre de canaux et d'augmenter la taille spatiale : https://www.geeksforgeeks.org/machine-learning/apply-a-2d-transposed-convolution-operation-in-pytorch/
        self.conv = Conv(out_channels*2,out_channels)
        
    def forward(self,x,x_enc):
        x = self.up(x) # ici, x et x_encodeur sont semblables
        x = torch.cat([x_enc,x],dim=1) #On concatene x et x_enc au niveau des canaux
        x = self.conv(x)
        return x
    
# Couche de sortie d'un Unet
class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
    def forward(self,x):
        return self.conv(x)

# Enfin le Unet
class Unet(nn.Module):
    def __init__(self,in_channels,enter_channels,nb_classes):
        super().__init__()
        # COnvolutions d'entrée
        self.enter = DoubleConv(in_channels,enter_channels)
        
        # Encodage 
        self.down1 = Down(enter_channels,enter_channels*2)
        self.down2 = Down(enter_channels*2,enter_channels*4)
        self.down3 = Down(enter_channels*4,enter_channels*8)
        self.down4 = Down(enter_channels*8,enter_channels*16)
        
        #décodage : taille d'origine de l'image avec enter_channels canaux
        self.up1 = Up(enter_channels*16,enter_channels*8)
        self.up2 = Up(enter_channels*8,enter_channels*4)
        self.up3 = Up(enter_channels*4,enter_channels*2)
        self.up4 = Up(enter_channels*2,enter_channels)
        
        # COnvolution de sortie, un masque par classe
        self.out = OutConv(enter_channels,nb_classes)
        
    def forward(self,x):
        x1 = self.enter(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x6 = self.up1(x5,x4)
        x7 = self.up2(x6,x3)
        x8 = self.up3(x7,x2)
        x9 = self.up4(x8,x1)
        
        out = self.out(x9)
        
        return out
        
        

def exemple_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet(3,64,10).to(device)
    train_loader = 0 # TODO
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    
    num_epoch = 5
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
    