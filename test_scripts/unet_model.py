
import torch
import torch.nn as nn
import torch.nn.functional as F

nch = [3, 16, 32, 64, 128, 256]

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(nch[0], nch[1], 3, 1, 1)
        self.conv1 = nn.Conv2d(nch[1], nch[2], 3, 1, 1)
        self.conv2 = nn.Conv2d(nch[2], nch[3], 3, 1, 1)
        ...
        self.conv5 = nn.Conv2d(nch[5], nch[5], 3, 1, 1)
        self.Upsample6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up6 = nn.Conv2d(nch[5], nch[4], 3, 1, 1)
        self.conv6 = nn.Conv2d(nch[4]*2, nch[4], 3, 1, 1)
        ...
        self.Upsample9 = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
        self.up9 = nn.Conv2d(nch[2], nch[1], 3, 1, 1)
        self.conv9 = nn.Conv2d(nch[1]*2, nch[1], 3, 1, 1)
        self.conv10 = nn.Conv2d(nch[1], 1, 1, 1, 0)
    def forward(self, X):
        conv_in = F.relu(self.conv_in(X))
        conv1 = F.relu(self.conv1(conv_in))
        pool1 = F.max_pool2d(conv1, 2)
        conv2 = F.relu(self.conv2(conv1))
        pool2 = F.max_pool2d(conv2, 2)
        ...
        conv5 = F.relu(self.conv5(conv4))
        up6 = F.relu(self.up6(self.Upsample6(conv5)))
        merge6 = torch.cat((conv4, up6), 1)
        conv6 = F.relu(self.conv6(merge6))
        ...
        up9 = F.relu(self.up9(self.Upsample9(conv8)))
        merge9 = torch.cat((conv1, up9), 1)
        conv9 = F.relu(self.conv9(merge9))
        conv10 = F.sigmoid(self.conv10(conv9))
        return conv10

