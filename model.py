import torch
import torch.nn as tnn

class ULayerDown(tnn.Module):
    """Downward layer of UNET module, contains a 2D convolution layer, an activation function and a Maxpooling layer.
Stores its forward outputs for referencing by upwards layers."""
    def __init__(self,channels_in,channels_out, **layer_kwargs):

        super().__init__()
        self.conv_layer = tnn.Conv2d(channels_in,channels_out,3,**layer_kwargs)
        self.activation = tnn.ReLU()
        self.scale_layer = tnn.MaxPool2d(2,padding=1)


        self.output = None

    def forward(self,X) :
            Y = self.conv_layer(X)
            Y = self.activation(Y)
            Y = tnn.Dropout(0.4)(Y)
            Y = self.scale_layer(Y)
            self.output = Y
            return Y


class ULayerUp(tnn.Module) :
    def __init__(self,single_feature_map_channels,n_classes,mask_shape,connected_nodes,_device="cuda", **layer_kwargs):
        """Upwards layer of UNET. handles concatenation and down/upsampling according to source layers and their levels.
        produces a side output corresponding to the mask predicted at this scale"""

        super().__init__()


        self.channels_per_fmap = single_feature_map_channels
        self.conv_layer = tnn.Conv2d(self.channels_per_fmap*(len(connected_nodes)+1), self.channels_per_fmap, 3,padding=1, **layer_kwargs)
        self.normed = tnn.BatchNorm2d(self.channels_per_fmap)
        self.activation = tnn.ReLU()
        self.scale_layer = tnn.Upsample(scale_factor=2)
        self.output = None

        self.device = _device

        self.prep_layers = []
        self.sources = connected_nodes

        self.side_mask_predictor = tnn.Sequential(tnn.Conv2d(self.channels_per_fmap,n_classes,3),
                                                  tnn.Sigmoid(),
                                                  tnn.Upsample(mask_shape)
                                                  )

        self.side_mask_output = None

        # add appropriate scaling/conv pipeline to the output of other layers

        for node in connected_nodes :
            self.prep_layers.append(tnn.Conv2d(node.conv_layer.out_channels,
                                                              self.channels_per_fmap,3,padding=1).to(self.device))

    def forward(self,X) :

        to_cat = [X]
        for i in range(len(self.sources )) :
            resized_output = tnn.Upsample(X.shape[-2:],mode="nearest")(self.sources[i].output)
            Xc = self.prep_layers[i](resized_output)
            to_cat.append(Xc)
        in_tensor = torch.cat(to_cat,1)
        Y = self.conv_layer(in_tensor)
        Y = self.normed(Y)
        Y = self.activation(Y)
        Y = tnn.Dropout(0.4)(Y)

        self.output = Y
        if  self.training and self.scale_layer is not None and self.side_mask_predictor is not None:
            Y = self.scale_layer(Y)
            self.side_mask_output = self.side_mask_predictor(Y)
        return Y

class ClassificationArm(tnn.Module) :
    """proposed classification module to make the model more robust to negative images by predicting class presence
    in images from the deepest encoding layer and multiplying output masks during training.

    """
    def __init__(self,in_channels,n_classes,input_size=25):
        super().__init__()

        pooled_size = input_size//2

        self.sequence = tnn.Sequential(tnn.Upsample([input_size,input_size]),
                                       tnn.Conv2d(in_channels,n_classes,3,padding=1),
                                       tnn.LeakyReLU(),
                                       tnn.AdaptiveMaxPool2d(pooled_size),
                                       tnn.Flatten(start_dim=1),
                                       tnn.Linear(n_classes*pooled_size*pooled_size,96),
                                       tnn.LeakyReLU(),
                                       tnn.Dropout(0.4),
                                       tnn.Linear(96,64),
                                       tnn.LeakyReLU(),
                                       tnn.Dropout(0.4),
                                       tnn.Linear(64,n_classes))
        self.sig_layer = tnn.Sigmoid()
    def forward(self,X):
        Y = self.sequence(X)
        return self.sig_layer(Y)





class UNet3plus(tnn.Module) :
    """UNET3+ autoencoder for semantic segmentation."""
    def __init__(self,
                 n_classes = 2,
                 in_channels = 3,
                 depth = 3,
                 first_output_channels = 8,
                 upwards_feature_channels = 16,
                 sideways_mask_shape = [200,200]):

        super().__init__()

        input_layer = ULayerDown(in_channels,first_output_channels)

        down_layers = [ULayerDown(first_output_channels*2**i,first_output_channels*2**(i+1)) for i in range(depth-1)]
        down_layers.append(ULayerDown(first_output_channels*2**(depth-1),upwards_feature_channels))
        down_layers.insert(0,input_layer)

        self.seqdown = tnn.Sequential(*down_layers)

        up_layers = []

        for i in range(depth) :

            up_layers.append(ULayerUp(upwards_feature_channels,
                            n_classes,
                            mask_shape = sideways_mask_shape,
                            connected_nodes= down_layers[:depth-1-i]+ up_layers ))

        self.sequp = tnn.Sequential(*up_layers)

        self.classifier = ClassificationArm(upwards_feature_channels,n_classes)

        self.last_layer = tnn.Conv2d(upwards_feature_channels,n_classes,3,padding=1)

        self._side_mask_shape = sideways_mask_shape

        self.presence_prediction = None

    def forward(self,X):

        Y = tnn.functional.normalize(X,dim=1)
        Y = self.seqdown(Y)
        Y = tnn.Upsample(scale_factor=2)(Y)
        self.presence_prediction = self.classifier(Y)

        Y = self.sequp(Y)

        for layer in self.sequp :
            layer.side_mask_output *= self.presence_prediction.unsqueeze(-1).unsqueeze(-1)

        Y = self.last_layer(Y)
        Y*= self.presence_prediction.unsqueeze(-1).unsqueeze(-1)
        Y = tnn.Upsample(X.shape[-2:])(Y)
        return tnn.Softmax(dim=1)(Y)


