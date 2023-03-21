import torch

def rollwindow(tensor,dim = 1,size = 5, step = None) :
    if step is None :
        step = size
    result = tensor.unfold(dim,size,step)
    result = result.unfold(dim+1,size,step)
    return result

def im_cov(I1,I2) :
    assert I1.nelement()==I2.nelement(), "tensors need to have the same size"
    Iflat, I2flat = I1.flatten().float(), I2.flatten().float()
    meanI = torch.mean(Iflat)
    meanI2 = torch.mean(I2flat)

    return torch.mean((Iflat-meanI)*(I2flat - meanI2))


def SSIM(image_1,image_2,beta=1,gamma=1,**rollwin_kwargs ) :
    """computes mean single-scale Structural SIMilarity index between two image-like tensors (CHW) using a sliding
    window."""

    #constants to avoid dividing by 0, given in MSSIM paper
    C1 = 1e-4
    C2 = 3e-4
    C3 = C2/2

    windows_1 = rollwindow(image_1,**rollwin_kwargs)
    windows_2 = rollwindow(image_1,**rollwin_kwargs)

    total_ssim = 0

    for i in range(windows_1.shape[1]) :
        for j in range(windows_1.shape[2]):

            m1,m2 = torch.mean(windows_1[:,i,j]),torch.mean(windows_2[:,i,j])
            s1,s2 = torch.std(windows_1[:,i,j]),torch.std(windows_2[:,i,j])

            C = (2*s1*s2 + C2) / (s1*s1 + s2*s2 + C2)
            S = (im_cov(windows_1[:,i,j],windows_2[:,i,j]) + C3) / (s1*s2+C3)

            total_ssim += C**beta * S**gamma

    return total_ssim/ (windows_1.shape[1]*windows_2.shape[2])














if __name__=="__main__":

    from torchvision.utils import save_image
    from torchvision.io import read_image
    import cv2
    I = read_image("rollwintest.jpg").unsqueeze(0)
    Ic = I.unfold(2,50,50)
    Ic = Ic.unfold(3,50,50)
    s = SSIM(Ic[0,:, 2, 0].float(),Ic[0,:, 2, 0].float())
    for i in range(6):
        for j in range(6):
            vig = Ic[0,:, i, j].permute(1, 2, 0)
            cv2.imwrite(f"./vig {i}_{j}.jpg", vig.numpy())