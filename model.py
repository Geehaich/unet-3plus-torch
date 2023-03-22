import torch
import torch.nn as tnn
from ClassArm import ClassificationArm




class UNodeDown(tnn.Module):
    """Downward layer of UNET module, contains a 2D convolution layer, an activation function and a Maxpooling layer.
Stores its forward outputs for referencing by upwards layers."""
    def __init__(self,channels_in,channels_out,level, **layer_kwargs):

        super().__init__()
        self.conv_layer = tnn.Conv2d(channels_in,channels_out,3,**layer_kwargs)
        self.activation = tnn.LeakyReLU()
        self.scale_layer = None
        self.level = level
        self.scale_layer = tnn.MaxPool2d(2,padding=1)


        self.output = None

    def forward(self,X) :
            Y = self.conv_layer(X)
            Y = self.activation(Y)
            Y = self.scale_layer(Y)
            self.output = Y
            return Y


class UNodeUp(tnn.Module) :
    def __init__(self,single_feature_map_channels,n_classes,mask_shape,level,connected_nodes, **layer_kwargs):
        """Upwards layer of UNET. handles concatenation and down/upsampling according to source layers and their levels.
        produces a side output corresponding to the mask predicted at this scale"""

        super().__init__()


        self.channels_per_fmap = single_feature_map_channels
        self.conv_layer = tnn.Conv2d(self.channels_per_fmap*(len(connected_nodes)+1), self.channels_per_fmap, 3,padding=1, **layer_kwargs)
        self.activation = tnn.LeakyReLU()
        self.scale_layer = tnn.Upsample(scale_factor=2)
        self.level = level
        self.output = None


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
                                                              self.channels_per_fmap,3,padding=1))

    def forward(self,X) :

        to_cat = [X]
        for i in range(len(self.sources )) :
            resized_output = tnn.Upsample(X.shape[-2:],mode="nearest")(self.sources[i].output)
            Xc = self.prep_layers[i](resized_output)
            to_cat.append(Xc)
        in_tensor = torch.cat(to_cat,1)
        Y = self.conv_layer(in_tensor)
        Y = self.activation(Y)

        self.output = Y
        if  self.training and self.scale_layer is not None and self.side_mask_predictor is not None:
            Y = self.scale_layer(Y)
            self.side_mask_output = self.side_mask_predictor(Y)
        return Y


class UNet3plus(tnn.Module) :

    def __init__(self,
                 n_classes = 1,
                 in_channels = 3,
                 first_output_channels = 8,
                 upwards_feature_channels = 16,
                 sideways_mask_shape = [200,200]):
        super().__init__()


        self.down_nodes = [UNodeDown(in_channels,first_output_channels,1),
                           UNodeDown(first_output_channels,first_output_channels*2,2),
                           UNodeDown(first_output_channels*2,first_output_channels*4,3),
                           UNodeDown(first_output_channels*4,upwards_feature_channels,4)]
        self.up_nodes = []
        for i in range(len(self.down_nodes)) :
            unode = UNodeUp(upwards_feature_channels,
                            n_classes,
                            level = len(self.down_nodes)-i-1,
                            mask_shape = 200,
                            connected_nodes=self.down_nodes[:len(self.down_nodes)-i]+self.up_nodes)
            self.up_nodes.append(unode)

        self.up_nodes[-1].scale_layer = None #last up node needs no upscaling, set to none to ignore in forward calls

        self.seqdown = tnn.Sequential(*self.down_nodes)
        self.sequp = tnn.Sequential(*self.up_nodes,tnn.Conv2d(upwards_feature_channels,n_classes,3,padding=1))

        self.Classifier = ClassificationArm(upwards_feature_channels,n_classes)

        self._side_mask_shape = sideways_mask_shape

    def forward(self,X):

        Y = self.seqdown(X)
        Y = tnn.Upsample(scale_factor=2)(Y)
        Y_presence_prediction = self.Classifier(Y)
        Y = self.sequp(Y)
        return tnn.Softmax(dim=1)(Y)



if __name__ == "__main__" :

    # down_nodes = [UNodeDown(3,8,1),UNodeDown(8,16,2),UNodeDown(16,32,3),UNodeDown(32,16,4)]
    # S = tnn.Sequential(*down_nodes)
    # up_nodes = []
    # for i in range(len(down_nodes)):
    #     unode = UNodeUp(16, len(down_nodes) - i - 1,
    #                     connected_nodes=down_nodes[:len(down_nodes) - i] + up_nodes)
    #     up_nodes.append(unode)

    X = torch.randn([3,3,50,50])
    model = UNet3plus(6)
    Y = model(X)
    # Y = S(X)
    # Y = tnn.Upsample(scale_factor=2)(Y)
    # Y = up_nodes[0](Y)
    # Y = up_nodes[1](Y)
    # Y = up_nodes[2](Y)


