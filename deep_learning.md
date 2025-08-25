
### Two ways of building a model, class and nn.Sequential()
See the two files `class_classification_model.py` and `sequential_classification_model.py`, they will be moved to FrequentModels later. <br>
Why are class names always in the parentheses after super when building a torch deep learning model? <br>
In Python 2, the syntax for calling a parent class's method (like `__init__`) using super() requires explicitly passing the current class name and the instance. This is a style that was required or common in Python 2, and is still valid in Python 3. Proper way in Python 3 is one of the two:
```python
super().__init__()
super(Classifier, self).__init__()
```

### Loss Functions
#### Classification losses
* nn.CrossEntropyLoss
  * Used for multi-class classification problems. It does not expect you to apply softmax before it because log_softmax is applied in it. 
  * Supports a weight argument (a 1D tensor) that assigns a weight to each class. Useful for imbalanced datasets.
  * Need to be instantiated as an object before using.  
* nn.functional.binary_cross_entropy
  * Sigmoid should be applied explicitly before the loss
  * Supports a weight argument to weight each element in the batch/sample-wise (not class weights).
* nn.functional.binary_cross_entropy_with_logits
  * Combines a Sigmoid layer and the BCE loss in one function. You don't have to apply sigmoid before passing logits here.
  * Supports the same weight argument as binary_cross_entropy for per-element weighting.
* Dice Loss
    * $\text{Dice} = \frac{2 \times |X \cap Y|}{|X| + |Y|}$
    * Dice Loss=1−Dice coefficient. 
    * Widely used in image segmentation tasks to measure the overlap between the predicted mask and the ground truth mask.
    * PyTorch does not include a built-in Dice Loss in its core torch.nn module.

#### Formula of Cross Entropy
$
\text{Binary Cross Entropy} = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$

Binary Cross Entropy Loss is given by:
$\text{Loss} = - \left[ y \log(p) + (1 - y) \log(1 - p) \right]$

Where:

- When \( y = 1 \): $ \text{Loss} = - \log(p)$
- When \( y = 0 \): $\text{Loss} = - \log(1 - p)$

Here,  
\(y\) is the true label (0 or 1),  
\(p\) is the predicted probability of class 1.

This figure shows the curves of Binary Cross Entropy and the derivation. 
![Derivation of the curve of cross entropy.png](figures/Derivation%20of%20the%20curve%20of%20cross%20entropy.png)

#### Examples of using these losses. 
```python
# This loss will be instantiated as an object first, no matter whether or not weights are applied
criterion = nn.CrossEntropyLoss(weight=weights)
loss = criterion(output, target)

loss = F.binary_cross_entropy(output, target, weight=weight)

loss = F.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)
```
#### Regression losses
* Mean Squared Error Loss (MSELoss)
  * Computes the average of squared differences between prediction and target. 
* Mean Absolute Error Loss (L1Loss)
  * Computes the average absolute difference between prediction and target. More robust to outliers than MSE.
* Huber Loss (SmoothL1Loss)
  * nn.functional.smooth_l1_loss
  * Combines MSE and MAE, less sensitive to outliers than MSE.

Examples of using these losses. 
```python
criterion = nn.MSELoss()
loss = criterion(input, target)

criterion = nn.L1Loss()
loss = criterion(input, target)

criterion = nn.SmoothL1Loss()
loss = criterion(input, target)
```
#### Triplet loss
PyTorch provides torch.nn.TripletMarginLoss which computes the triplet loss given anchor, positive, and negative embeddings. It measures the relative distance between these samples, encouraging the anchor to be closer to the positive than to the negative by at least a margin. Formula for each triplet is: **max(∥a−p∥−∥a−n∥+margin,0)**. <br>
Code for using this loss:
```python
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
loss = triplet_loss(anchor, positive, negative)
```
The parameter `p` determines which Lp norm to use when computing distances. By default, p=2, which means the `Euclidean distance` (L2 norm) is used. If you set p=1, it would compute the `Manhattan distance` (L1 norm).

#### Focal Loss
Analysis of Class Imbalance and Hard/Easy Examples. 
* #positive samples >> #negative samples (class imbalance).
* #easy samples >> #hard samples (easy negative samples dominate).

Faster RCNN solved the above two problems by:
* Set # positive samples : # negative samples = 1:3
* RPN picks out roughly 1000 positive samples, i.e. dropping out the
dominated easy negative samples.

Focal loss for binary classification:

$
L(y, \hat{p}) = -\alpha \, y (1 - \hat{p})^\gamma \log(\hat{p}) - (1 - y) \, (1 - \alpha) \hat{p}^\gamma \log(1 - \hat{p})
$

where:
- y is the ground-truth label 0 or 1.
- $\hat{p}$ is the model's estimated probability for class 1.
- $\alpha$ is the weighting factor (usually for the positive class), usually this is inverse class frequency or treated as hyper-parameter. 
- $\gamma$ is the focusing parameter.

For (y = 1): $L(1, \hat{p}) = -\alpha \, (1-\hat{p})^\gamma \log(\hat{p})$<br>
For (y = 0): $L(0, \hat{p}) = -(1 - \alpha) \hat{p}^\gamma \log(1 - \hat{p})$<br>
In summary, for focal loss, losses of cases where the errors are larger are given larger weights, and $\alpha$ solves the `class imbalance` issue. 

# To page 167 of detection_2stage_1stage. 

#### Formulas for these loss functions. 

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
* YOLO
* Faster-RCNN
* SSD
* LLM

### Modern ConvNets
* **GoogLeNet**. The development of an Inception Module that dramatically reduced the number of parameters in the network.
* **VGGNet**. 
* **ResNet**. It features special skip connections and a heavy use of batch normalization.
* **Inception-V4**. Inception+ResNet. Seven (JulyEDU) ConvNets Principles
* **MobileNet**. 

### NMS
Non-Maximum Suppression. <br>
#### The problem of NMS and the solution soft-NMS
In an image there are two very close confident horse detections, which have scores of 0.95 and 0.8 respectively. The one with score 0.8 will be suppressed to zero when picking the one with 0.95. The solution is assigning it a slightly lower score of 0.4. This solution is called soft-NMS. 

#### Pseudo code for NMS and soft-NMS
```python
#B is the list of initial detection boxes
#S contains corresponding detection scores
#Nt is the NMS threshold
def NMS(B = [b1, ..., bn], S = [s1, ..., sn], Nt):
    D = []
    while len(B) > 0:
        m = argmax(S)
        M = b[m]
        D = D.append(M)
        for bi in B:
            if iou(bi, M) > Nt: # code for NMS
                B = B - bi, S = S - si
            #si = si*f(iou(M, bi)) #code for soft-NMS
    return D
```
Another alternative to NMS is `adaptive NMS`, where the threshold varies as the training goes on. See July slides. 

### Model validation
#### IOU
#### Dice Coefficient
$\text{Dice} = \frac{2 \times |X \cap Y|}{|X| + |Y|}$
#### Recall Precision and Recall
$\text{Precision} = \frac {TP}{TP+FP}$<br>
$\text{Recall} = \frac {TP}{TP+FN}$<br>

Precision and Recall curve

### Instance Models
#### RCNNs
* R-CNN, fast R-CNN, faster R-CNN. <br>
* Roi Pooling is applied on fast R-CNN. <br>
* RPN Region Proposal Network is applied on faster R-CNN. <br>
* The RoIAlign layer is designed to fix the location misalignment caused by quantization in the RoI pooling. Bilinear interpolation is applied. Faster RCNN originally used roi pooling, and later Roi align was introduced in Mask RCNN, and later implementations of faster RCNN uses Roi Align. <br>
* In faster RCNN, each anchor corresponds 3 different scales: {128, 256, 512} and 3 different **aspect ratios**: {1:1, 1:2, 2:1}.

The Pseudo-code of Fast R-CNN:
```python
feature maps = process(image)
ROIs = region proposal(image) # selective search running on CPU is very slow! 
for ROI in ROIs
    patch = roi pooling(feature maps, ROI)
    results = detector2(patch)
```
#### YOLOs
The most salient feature of v3 is that it makes detections at three
different scales. YOLO is a fully convolutional network and its eventual
output is generated by applying a 1x1 kernel on a feature map. In
YOLO-V3, the detection is done by applying 1x1 detection kernels on
feature maps of three different sizes at three different places in the
network. <br>
**See YOLO-v3 Network Architecture.**
* k-means clustering algorithm was used for selecting the anchor sizes
* Upsampling is applied in YOLO-v3



### Regularization
Regularization is a common technique to prevent model learning from overfitting. 
* L1 Regularization: Penalizes the **absolute value** of coefficients, **encouraging sparse models** where unnecessary features are set to zero. 
  * $Loss_{L1} = Loss + \lambda \sum_{i=1}^{n} |W_i|$
* L2 Regularization: Penalizes the __square of coefficients__, **encouraging smaller weights** spread more evenly across predictors.
  * $Loss_{L2} = Loss + \lambda \sum_{i=1}^{n} W_i^2$
* Elastic Net: Combines L1 and L2 penalties to integrate both benefits.
* Dropout: Randomly drops units during training.
* Early Stopping: Stops model training when further improvement on validation data ceases.

### Activation functions
* **Sigmoid** 
  * $\sigma(x) = \frac{1}{1+ e^{-x}}$
* **Tanh (Hyperbolic Tangent)** 
  * $\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$
  * Sigmoids and Tanhs may saturate and kill gradients!
* **ReLU (Rectified Linear Units)** 
  * f(x) = max(0, x)
  * Accelerate the convergence significantly.
  * More efficient implementation compared with exponencials in Sigmoid/Tanh.
  * ReLU units can be “dead” during training.
* **Leaky ReLU** 
  * f(x) = max(a*x, x)
  * Fixes the “dying ReLU” problem. 

### Output Size and  Receptive Field of a CNN Neuron
* Output size of a CNN neuron: OutputSize = (W+2P-F)/S + 1
  * W: Input size. P: Padding. F: Kernel size. S: Stride. 
  * If the number is not an integer, then the strides are set incorrectly. 
output_field_size = (input_field_size-kernel_size + 2*padding)/stride + 1
input field size = (output field size-1) * stride-2*padding + kernel size


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
Structures for encoder and decoder. 

## To do list
* SWIN architecture
* ResNet
* GoogleNet

