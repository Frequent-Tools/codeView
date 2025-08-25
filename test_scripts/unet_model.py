
import torch.nn as nn
import torch.nn.functional as F
from params import *

#n_ch = [3, 32, 64, 128, 256, 256]
n_ch = [3, 16, 32, 64, 128, 256]

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv_in = nn.Conv2d(n_ch[0], n_ch[1], 3, 1, padding=1)
        self.conv1a = nn.Conv2d(n_ch[1], n_ch[1], 3, 1, padding=1)
        self.conv1b = nn.Conv2d(n_ch[1], n_ch[1], 3, 1, padding=1)
        self.conv2a = nn.Conv2d(n_ch[1], n_ch[2], 3, 1, padding=1)
        self.conv2b = nn.Conv2d(n_ch[2], n_ch[2], 3, 1, padding=1)
        self.conv3a = nn.Conv2d(n_ch[2], n_ch[3], 3, 1, padding=1)
        self.conv3b = nn.Conv2d(n_ch[3], n_ch[3], 3, 1, padding=1)
        self.conv4a = nn.Conv2d(n_ch[3], n_ch[4], 3, 1, padding=1)
        self.conv4b = nn.Conv2d(n_ch[4], n_ch[4], 3, 1, padding=1)
        self.conv5a = nn.Conv2d(n_ch[4], n_ch[5], 3, 1, padding=1)
        self.conv5b = nn.Conv2d(n_ch[5], n_ch[5], 3, 1, padding=1)
        self.upsample6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up6 = nn.Conv2d(n_ch[5], n_ch[4], 3, 1, padding = 1)
        self.conv6a = nn.Conv2d(n_ch[4]*2, n_ch[4], 3, 1, padding=1)
        self.upsample7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up7 = nn.Conv2d(n_ch[4], n_ch[3], 3, 1, padding=1)
        self.conv7a = nn.Conv2d(n_ch[3]*2, n_ch[3], 3, 1, padding=1)
        self.upsample8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up8 = nn.Conv2d(n_ch[3], n_ch[2], 3, 1, padding=1)
        self.conv8 = nn.Conv2d(n_ch[2]*2, n_ch[2], 3, 1, padding=1)
        self.upsample9 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up9 = nn.Conv2d(n_ch[2], n_ch[1], 3, 1, padding=1)
        self.conv9 = nn.Conv2d(n_ch[1]*2, n_ch[1], 3, 1, padding=1)
        self.conv10 = nn.Conv2d(n_ch[1], 1, 1, 1, padding=0)

    def forward(self, imgs):
        conv_in = F.relu(self.conv_in(imgs))
        conv1a = F.relu(self.conv1a(conv_in))
        conv1b = F.relu(self.conv1b(conv1a))
        pool1 = F.max_pool2d(conv1b, 2)
        conv2a = F.relu(self.conv2a(pool1))
        conv2b = F.relu(self.conv2b(conv2a))
        pool2 = F.max_pool2d(conv2b, 2)
        conv3a = F.relu(self.conv3a(pool2))
        conv3b = F.relu(self.conv3b(conv3a))
        pool3 = F.max_pool2d(conv3b, 2)
        conv4a = F.relu(self.conv4a(pool3))
        conv4b = F.relu(self.conv4b(conv4a))
        pool4 = F.max_pool2d(conv4b, 2)
        conv5a = F.relu(self.conv5a(pool4))
        conv5b = F.relu(self.conv5b(conv5a))
        up6 = F.relu(self.up6(self.upsample6(conv5b)))
        merge6 = torch.cat((conv4b, up6), 1)
        conv6a = F.relu(self.conv6a(merge6))
        up7 = F.relu(self.up7(self.upsample7(conv6a)))
        merge7 = torch.cat((conv3b, up7), 1)
        conv7a = F.relu(self.conv7a(merge7))

        up8 = F.relu(self.up8(self.upsample8(conv7a)))
        merge8 = torch.cat((conv2b, up8), 1)
        conv8a = F.relu(self.conv8(merge8))
        up9 = F.relu(self.up9(self.upsample9(conv8a)))
        merge9 = torch.cat((conv1b, up9), 1)
        conv9a = F.relu(self.conv9(merge9))
        conv10 = F.sigmoid(self.conv10(conv9a))
        output = conv10
        return output
