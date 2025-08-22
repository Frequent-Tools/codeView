
## 面试复习库
这是一个面试准备素材的工程，所有面试将可能涉及到的知识点都会放在这里，在面试前花短时间复习。

### Two ways of building a model, class and nn.Sequential()

This is Perplexity's answer for building a model with `nn.Sequential`. 
```
import torch
import torch.nn as nn

# Build a classification model with nn.Sequential
model = nn.Sequential(
    nn.Linear(20, 64),      # Input: 20 features, Output: 64 hidden units
    nn.ReLU(),
    nn.Linear(64, 10)       # Output: 10 classes (no activation for raw scores)
)

# For binary classification, use nn.Sigmoid at the end; for multi-class, use nn.Softmax(dim=1) if you want probabilities
# For training, use nn.CrossEntropyLoss (which expects raw scores, NOT softmax outputs)
```



* Unet model
* Two ways of building model: class and 






