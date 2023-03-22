import torch
import torch.nn as tnn

class ClassificationArm(tnn.Module) :

    def __init__(self,in_channels,n_classes,input_size=25):
        super().__init__()

        self.sequence = tnn.Sequential(tnn.Upsample([input_size,input_size]),
                                       tnn.Conv2d(in_channels,n_classes,3,padding=1),
                                       tnn.LeakyReLU(),
                                       tnn.AdaptiveMaxPool2d(2),
                                       tnn.Flatten(start_dim=1),
                                       tnn.LazyLinear(96),
                                       tnn.LeakyReLU(),
                                       tnn.Dropout(0.4),
                                       tnn.Linear(96,64),
                                       tnn.LeakyReLU(),
                                       tnn.Dropout(0.4),
                                       tnn.Linear(64,n_classes),
                                       tnn.Sigmoid())
    def forward(self,X):
        return self.sequence(X)

