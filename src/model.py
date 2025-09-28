import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

IMAGE_SIZE = 256

class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        backbone = resnet50(weights=weights)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x0, x2, x3, x4, x5

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec1 = ConvBlock(1024 + 1024, 1024)
        
        self.up2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec2 = ConvBlock(512 + 512, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(256 + 256, 256)
        
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = ConvBlock(128 + 64, 128)

        self.up5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec5 = ConvBlock(64, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x0, x2, x3, x4, x5):
        d1 = self.up1(x5)
        d1 = torch.cat([d1, x4], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x3], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([d4, x0], dim=1)
        d4 = self.dec4(d4)
        
        d5 = self.up5(d4)
        d5 = self.dec5(d5)
        
        out = self.final(d5)
        return out

class MultiTaskShared(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.decoder = UNetDecoder()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.cls_head = nn.Linear(2048, num_classes)

    def forward(self, x):
        x_resized_for_model = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        x0, x2, x3, x4, x5 = self.encoder(x_resized_for_model)
        
        seg_logits = self.decoder(x0, x2, x3, x4, x5)
        
        gap = self.pool(x5).flatten(1)
        cls_logits = self.cls_head(gap)
        
        return seg_logits, cls_logits