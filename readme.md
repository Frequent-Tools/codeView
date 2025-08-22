
## 面试复习库
这是一个面试准备素材的工程，所有面试将可能涉及到的知识点都会放在这里，在面试前花短时间复习。

### Two ways of building a model, class and nn.Sequential()
See the two files `class_classification_model.py` and `sequential_classification_model.py`, they will be moved to FrequentModels later. <br>
Why are class names always in the parentheses after super when building a torch deep learning model? <br>
In Python 2, the syntax for calling a parent class's method (like `__init__`) using super() requires explicitly passing the current class name and the instance. This is a style that was required or common in Python 2, and is still valid in Python 3. Proper way in Python 3 is one of the two:
```python
super().__init__()
super(Classifier, self).__init__()
```

### Loss Functions
```python
import torch.nn as nn
nn.CrossEntropyLoss
nn.functional.binary_cross_entropy
nn.functional.binary_cross_entropy_with_logits
nn.functional.smooth_l1_loss
Triplet loss
# Equations for these loss functions. 
```
### optimizer (e.g., Adam or SGD)
Some differeces listed by Perplexity.
* Adam handles adaptive learning rates per parameter, while SGD uses a global learning rate. 
* Use Adam for complex models (like NLP, Transformers), or when fast convergence is important.
* Use SGD (preferably with momentum) for training to achieve better final accuracy/generalization, especially on vision tasks if you can afford longer training and tuning.

### Frequent models
* Classification
  * Built based on a class or on nn.Sequential()
* Unet
* Transformer for translation
* VIT
* SWIN



### How to calculate the receptive field


### Write the training loop
```python
import torch
from EncEncCatEnc_model import Transformer
from dataloader import Dataset, Dataloader

device = torch.device('cuda')
model = Transformer(...).to(device)

train_dataset = Dataset(dirs, img_w, img_h)
train_loader = Dataloader(train_dataset, batch_size)

model.train()
optim = torch.optim.Adam(model.parameters(), lr=0.1)
epoch_num = 50
for epoch in range(epoch_num):
    for batch_idx, batch_data in enumerate(train_loader):
        batch_imgs, batch_masks = batch_data
        outputs = model(batch_imgs)
        loss = getLoss(outputs, batch_masks)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f'epoch: {epoch+1}/{epoch_num}, batch: {batch_idx+1}/{len(train_loader)}, loss: {loss.item()}')
    torch.save(model.state_dict(), '.../EncEncCatEnc_%03d.pkl'%(epoch+1))
```

### More flexible learning rate
* How to schedule learning rate

* How to freeze part of the Model

* How to set different lr for different part of the model

### Convert a pth to onnx


### Write a dataloader
This is a dataloader given by Perplexity, simplified by me.<br>
For implementing my own Dataset and Dataloader, which have more flexibility, see the examples in `EncHintNailsSeg`. 
```python
from torch.utils.data import Dataset, DataLoader

class CSVLoader(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return features, label

dataset = CSVLoader('data.csv')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch_idx, (features, labels) in enumerate(dataloader):
    print(...)
``` 

### Transformer for translation







