
import torch
from Unet_model import Unet
import torch.optim as optim
import torch.nn.functional as F
from dataloader import dataset, dataloader

device = torch.device("cuda")
model = Unet().to(device)
model.train()

trainset = dataset("train_dir")
trainloader = dataloader(trainset, batch_size=4, shuffle=True)
n_epochs = 10

optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch_idx in range(n_epochs):
    for batch_idx, batch_data in enumerate(dataloader):
        batch_imgs, batch_masks = batch_data
        segs = model(batch_imgs)
        loss = F.binary_cross_entropy(segs, batch_masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
