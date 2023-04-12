from model import *
from loss import compound_unet_loss
import os
from tqdm import tqdm

import torch
import torch.nn as tnn
from torch.utils.data import Dataset,DataLoader

LOSS_EXPONENTS_BETA = [0.0448,0.2856,0.3001,0.2363,0.1333] #exponents used in MS-SSIM loss function in the UNET3+ paper
LOSS_EXPONENTS_GAMMA = [0.0448,0.2856,0.3001,0.2363,0.1333] #results were found in MS SIM paper for a specific experiment, maybe not adapted to all datasets

def train(model : UNet3plus,
          data_train,
          data_test=None,
          batch_size = 8,
          epochs = 10 ,
          early_stop_threshold = 1e-4,
          early_stop_patience = 5,
          input_image_size = 256,
          save_model_directory = None
          ) :

    dl_train, dl_test = DataLoader(data_train,batch_size), DataLoader(data_test,batch_size)

    optimizer_image = torch.optim.Adam(model.parameters())
    optimizer_classification_arm = torch.optim.Adam(model.classifier.parameters())

    best_test_loss = torch.inf
    patience = early_stop_patience

    classifier_loss = tnn.BCELoss()

    resizer = tnn.Upsample(input_image_size)


    for epoch in range(epochs):

        print(f"\nEPOCH {epoch+1} / {epochs} :  ")
        train_loss_class = 0
        train_loss_mask = 0
        test_loss_mask = 0
        test_loss_class = 0

        with tqdm(dl_train) as train_batches_tqdm :

            i = 1 #count batches for avg
            for (image,mask,class_presence) in train_batches_tqdm :

                optimizer_image.zero_grad()
                optimizer_classification_arm.zero_grad()
                model.classifier.requires_grad_ = False
                im_resized = resizer(image)
                mask_resized = resizer(mask)

                mask_output = model(im_resized)

                # get predicted masks at different levels of upscaler and resize to same dimensions as ground truth
                # masks have additional background class as first channel, remove it for loss
                preds = [resizer(inter_mask)[:, 1:, ...] for inter_mask in
                         [layer.side_mask_output for layer in model.sequp[:-1]]]
                preds += [mask_output[:, 1:, ...]]


                image_loss= compound_unet_loss(preds,mask_resized[:, 1:, ...],LOSS_EXPONENTS_BETA,LOSS_EXPONENTS_GAMMA)
                train_loss_mask += image_loss.item()
                image_loss.backward(retain_graph=True)

                optimizer_image.step()

                #independent backward pass on classifier
                model.classifier.requires_grad = True

                model.seqdown.requires_grad = False
                model.sequp.requires_grad = False

                class_loss = classifier_loss(model.presence_prediction[:,1:],class_presence[:,1:])
                train_loss_class += class_loss.item()
                class_loss.backward()
                optimizer_classification_arm.step()

                model.seqdown.requires_grad = True
                model.sequp.requires_grad = True


                train_batches_tqdm.set_description(f" TRAIN --- Seg. Loss :{train_loss_mask/i:.5f} | Class.Loss :{train_loss_class/i:.5f}")

                i+=1

        if dl_test :

            with tqdm(dl_test) as test_batches_tqdm:


                i = 1  # count batches for avg
                for (image, mask, class_presence) in test_batches_tqdm:

                    im_resized = resizer(image)
                    mask_resized = resizer(mask)
                    mask_output = model(im_resized)

                    #get predicted masks (skipping first background class) at different levels of upscaler and resize to same dimensions as ground truth

                    preds = [resizer(inter_mask)[:,1:,...] for inter_mask in [layer.side_mask_output for layer in model.sequp[:-1]]]
                    preds += [mask_output[:,1:,...]]

                    image_loss = compound_unet_loss(preds, mask_resized[:,1:,...], LOSS_EXPONENTS_BETA, LOSS_EXPONENTS_GAMMA)
                    presence_loss = classifier_loss(model.presence_prediction, class_presence)

                    test_loss_class += presence_loss.item()
                    test_loss_mask += image_loss.item()

                    test_batches_tqdm.set_description(
                        f" TEST ---  Seg. Loss :{test_loss_mask / i:.5f} | Class.Loss :{test_loss_class / i:.5f}")

                    i += 1

            if test_loss_mask + early_stop_threshold < best_test_loss :

                patience == early_stop_patience
                best_test_loss = test_loss_mask
                if save_model_directory is not None :
                    filename = save_model_directory+"/best.pt"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    torch.save(model,filename)


            else :
                patience -= 1
                if patience == 0 :
                    print(f"Early stopping at epoch {epoch+1} : {early_stop_patience} epochs without improvement")
                    if save_model_directory is not None:
                        filename = save_model_directory + "/last.pt"
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        torch.save(model, filename)
                    return
        else :
            if epoch %5 == 0 :
                filename = save_model_directory + "/last_5th_epoch.pt"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                torch.save(model, filename)

    if save_model_directory is not None:
        filename = save_model_directory + "/last.pt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(model, filename)
    return




