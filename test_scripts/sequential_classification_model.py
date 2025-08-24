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



