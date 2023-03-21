import torch
import torch.nn as tnn

class ClassificationArm(tnn.Module) :

    def __init__(self,in_channels,n_classes):
        super().__init__()

        self.sequence = tnn.Sequential(tnn.Conv2d(in_channels,n_classes,3,padding=1),
                                       tnn.LeakyReLU(),
                                       tnn.AdaptiveMaxPool2d(2),
                                       tnn.flatten(),
                                       tnn.LazyLinear(64),
                                       tnn.LeakyReLU(),
                                       tnn.Dropout(),
                                       tnn.Linear(64,n_classes),
                                       tnn.Sigmoid())
    def forward(self,X):
        return self.sequence(X)

