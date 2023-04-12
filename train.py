import torch.nn as tnn

from model import UNet3plus
from dataset import *
from trainfunc import train
from argparse import ArgumentParser
from torch.utils.data import random_split

parser =  ArgumentParser(
                    prog='UNET+3',
                    description='Train a UNET3+ model on a train and test set using COCO-style labels',
                    )

parser.add_argument("-training_set",required = False,help="Path to directory containing the training folder containing_annotations.json file describing mask polygons")
parser.add_argument("--test_set",default = None, help = "Path to test dataset. or use --train_test_split to split training set")
parser.add_argument("--train_test_split",default = None, help="if test_set is empty, enter a number in the 0-100 range to split training set")

parser.add_argument("--img_size",default=256, type=int,help = "size training images are resized to")
parser.add_argument("--epochs",default=20,type=int)
parser.add_argument("--batch_size",default=8,type=int)
parser.add_argument("--device",default=0, help= "device used for training. defaults to cuda:0. -1 for cpu",type=int)
parser.add_argument("--early_stopping_patience",default=5,type=int)
parser.add_argument("--early_stopping_threshold",default=1e-4,type=float)

parser.add_argument("--save_model_directory",default='./')
parser.add_argument("--model_first_output_channels",default= 16,type=int)
parser.add_argument("--model_depth",default= 4,type=int)
parser.add_argument("--model_input_channels",default= 3,type=int)
parser.add_argument("--model_up_feature_channels",default= 32 ,help = "number of channels of decoder stage outputs, and how many channels their input are reducted to.",type=int)
parser.add_argument("--model_side_mask_size",default= 256 ,help = "size of side mask outputs of the decoder stage used during training.",type=int)



args = parser.parse_args()


device = torch.device(args.device) if args.device >= 0  else torch.device("cpu")
dataset = COCODataset(args.training_set,
                         _device= args.device,
                         image_shape= [args.img_size]*2)
Dset_test = None
Dset_train = dataset
if args.test_set :
    Dset_test = COCODataset(args.test_set)
else :
    if args.train_test_split:
        ratio = int(args.train_test_split)/100
        Dset_train,Dset_test = random_split(dataset,[1-ratio,ratio])

model = UNet3plus(in_channels=args.model_input_channels,
                  n_classes=len(Dset_train.categories),
                  depth= args.model_depth,
                  first_output_channels= args.model_first_output_channels,
                  upwards_feature_channels=args.model_up_feature_channels,
                  sideways_mask_shape=[args.model_side_mask_size,args.model_side_mask_size]).to(device)

Y = model(dataset[0][0].unsqueeze(0))
train(model,
      Dset_train,
      Dset_test,
      epochs=args.epochs,
      batch_size=args.batch_size,
      input_image_size=args.img_size,
      save_model_directory=args.save_model_directory,
      early_stop_patience= args.early_stopping_patience,
      early_stop_threshold= args.early_stopping_threshold)


