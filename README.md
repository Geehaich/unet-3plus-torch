# UNET+3 for semantic segmentation

An implementation of the full-scale connected UNET3+ architecture as described by Huang et al. in 2020 in Pytorch.

UNET3+ is a UNET variant with a higher amount of skip connections and a classification module running on the encoder output to reduce the amount of false positives.
First tests on small specific datasets seem to confirm the paper's results, precision similar to regular UNET despite overall lower number of parameters.

## Contents of the repo

* A flexible constructor allowing for model variations in terms of depth and width
* implementation of loss functions described in the research paper*
* training functions and scripts for training on datasets with COCO-styled annotations

## Caveats
 * MS-SSIM loss involves a fractional power of image covariance, the paper gave no indication of the value of the loss function in cases where cov < 0.
 We ignore such cases during training, which doesn't seem to impact the model's ability to converge.

