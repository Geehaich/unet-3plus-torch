import torch
from torchvision.ops import sigmoid_focal_loss

def rollwindow(tensor,dim = 2,size = 25, step = None) :
    """uses tensor.unfold to return a tensor containing sliding windows over the source.
    windows don't overlap by default, use the step parameter to change it."""
    if step is None :
        step = size
    result = tensor.unfold(dim,size,step)
    result = result.unfold(dim+1,size,step)
    return result

def batch_im_cov(I1,I2) :
    assert I1.nelement()==I2.nelement(), "tensors need to have the same size"

    Iflat, I2flat = I1.flatten(start_dim=1).float(), I2.flatten(start_dim=1).float()
    meanI = torch.mean(Iflat,dim=1)
    meanI2 = torch.mean(I2flat,dim=1)

    return torch.mean((Iflat-meanI.unsqueeze(-1))*(I2flat - meanI2.unsqueeze(-1)),dim=1)


def SSIM(image_1,image_2,beta=1,gamma=1,**rollwin_kwargs ) :
    """computes mean single-scale Structural SIMilarity index between two image-like tensors (NCHW) using a sliding
    window."""

    #constants to avoid dividing by 0, given in MSSIM paper
    C1 = 1e-4
    C2 = 9e-4

    windows_1 = rollwindow(image_1,**rollwin_kwargs)
    windows_2 = rollwindow(image_2,**rollwin_kwargs)

    total_ssim = torch.zeros(image_1.shape[0],device=image_1.get_device())

    for i in range(windows_1.shape[2]) :
        for j in range(windows_1.shape[3]):

            m1,m2 = torch.mean(windows_1[:,:,i,j],dim=(1,2,3)),torch.mean(windows_2[:,:,i,j],dim=(1,2,3))
            s1,s2 = torch.std(windows_1[:,:,i,j],dim=(1,2,3)),torch.std(windows_2[:,:,i,j],dim=(1,2,3))

            C = (2*m1*m2 + C1) / (m1*m1 + m2*m2 + C1)
            S = (2*batch_im_cov(windows_1[:,:,i,j],windows_2[:,:,i,j]) + C2) / (s1*s1+s2*s2+C2)

            window_sim = C**beta * S**gamma
            if not torch.any(window_sim.isnan()) :
                total_ssim += window_sim

    return total_ssim/ (windows_1.shape[2]*windows_2.shape[3]+0.0001)


def IoU_loss(pred :torch.Tensor,targ :torch.Tensor) :
    """Intersection over Union. basic loss"""
    pred_flat,targ_flat = pred.flatten(start_dim=1),targ.flatten(start_dim=1)
    intersection = torch.sum(pred_flat*targ_flat)
    union = pred_flat.sum()+targ_flat.sum() - intersection
    return 1- (intersection+0.1)/(union+0.1)


def MS_SSIM_loss (pred_scales,target,betas = [1],gammas = [1]) :
    """multi-scale similarity loss computed from the product of losses at similar scales and a final target image.
Typically used to compare the intermediary outputs of the decoder branch and the ground truth mask."""
    SSIM_product= 1
    for i in range(len(pred_scales)) :
        pred = pred_scales[i][:,1:,...]
        _beta = betas[min(i,len(betas)-1)]
        _gamma = gammas[min(i,len(gammas)-1)]
        SSIM_product *= SSIM(pred,target[:,1:,...],_beta,_gamma)

    return 1-SSIM_product


def compound_unet_loss(pred_scales,target,betas,gammas) :

    focal_loss = sigmoid_focal_loss(pred_scales[-1],target,reduction="mean") #pixel level loss
    iou_loss  = IoU_loss(pred_scales[-1],target) #image level
    ms_loss = MS_SSIM_loss(pred_scales,target,betas,gammas)  #patch level
    ms_loss = torch.mean(ms_loss)

    return  ms_loss+focal_loss+iou_loss

