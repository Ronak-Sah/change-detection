import torch.nn as nn
import torch


from src.components.models.ConvolutionLayer import ConvBlock
class FCSiamDiff(nn.Module):
    def __init__(self,in_ch=1,num_classes=2):
        super().__init__()
        # ----- Encoder(shared) -----

        self.enc1=ConvBlock(in_ch,32)
        self.enc2=ConvBlock(32,64)
        self.enc3=ConvBlock(64,128)

        self.pool=nn.MaxPool2d(2,2)

        # ---------- Decoder ----------
        self.up2=nn.ConvTranspose2d(128,64,2,stride=2)
        self.dec2=ConvBlock(64,64)

        self.up1=nn.ConvTranspose2d(64,32,2,stride=2)
        self.dec1=ConvBlock(32,32)

        self.classifier = nn.Conv2d(32, num_classes, 1)

    def forward(self,before,after):
        # ----- Encoder -----
        
        b1 = self.enc1(before)
        b2 = self.enc2(self.pool(b1))
        b3 = self.enc3(self.pool(b2))

        a1 = self.enc1(after)
        a2 = self.enc2(self.pool(a1))
        a3 = self.enc3(self.pool(a2))

        # ----- Feature difference -----
        d3 = torch.abs(b3 - a3)

        # ----- Decoder -----
        x = self.up2(d3)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        out = self.classifier(x)
        return out