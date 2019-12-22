import numpy as np
import torch
from torch import tensor
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torchvision import transforms

def init_fft_buf(size):
    sd = 0.01
    img_buf = np.random.normal(size=(1, 3, size, size//2 + 1, 2), scale=sd).astype(np.float32)
    spectrum_t = tensor(img_buf).float().cuda()
    return spectrum_t
    
def fft_to_rgb(t, d=0.5, decay_power=1, fft_magic=4.0, **kwargs):
    size = t.shape[-3]

    fy = np.fft.fftfreq(size,d=d)[:,None]
    fx = np.fft.fftfreq(size,d=d)[: size//2 + 1]
    freqs = (np.sqrt(fx * fx + fy * fy))
    scale = 1.0 / np.maximum(freqs, 1.0 / size) ** decay_power
    scale = tensor(scale).float()[None,None,...,None].cuda()
    scale *= size
    t = scale * t

    image_t = torch.irfft(t,2,signal_sizes=(size,size))
    image_t = image_t / fft_magic

    return image_t

def lucid_colorspace_to_rgb(t):
    color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                             [0.27, 0.00, -0.05],
                                             [0.27, -0.09, 0.03]]).astype(np.float32)
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

    t_flat = t.permute(0,2,3,1)
    color_correlation_normalized = tensor(color_correlation_svd_sqrt / max_norm_svd_sqrt).cuda()
    t_flat = torch.matmul(t_flat, color_correlation_normalized.T)
    t = t_flat.permute(0,3,1,2)
    return t

def image_buf_to_rgb(img_buf, jitter, **kwargs):
    img = img_buf.detach()
    img = fft_to_rgb(img, **kwargs)
    size = img.shape[-1]
    x_off,y_off = jitter//2,jitter//2
    img = lucid_colorspace_to_rgb(img)
    img = torch.sigmoid(img)
    img = img[:,:,x_off:x_off+size-jitter,y_off:y_off+size-jitter] # jitter
    img = img.squeeze()    
    return img
    
def show_rgb(img, label=None, ax=None, dpi=25, **kwargs):
    plt_show = True if ax == None else False
    if ax == None: _, ax = plt.subplots(figsize=(img.shape[1]/dpi,img.shape[2]/dpi))
    x = img.cpu().permute(1,2,0).numpy()
    ax.imshow(x)
    ax.axis('off')
    ax.set_title(label)
    if plt_show: plt.show()

def visualize_feature(model, layer, feature,
                      size=200, jitter=25,
                      steps=2000, lr=0.05,
                      debug=False, frames=10, show=True, **kwargs):
    img_buf = init_fft_buf(size+jitter)
    img_buf.requires_grad_()
    opt = torch.optim.Adam([img_buf], lr=lr)

    hook_out = None
    def callback(m, i, o):
        nonlocal hook_out
        hook_out = o
    hook = layer.register_forward_hook(callback)
    
    for i in range(1,steps+1):
        x_off, y_off = int(np.random.random()*jitter),int(np.random.random()*jitter)
        img = fft_to_rgb(img_buf, **kwargs)
        img = img[:,:,x_off:x_off+size+1,y_off:y_off+size+1] # jitter
        img = lucid_colorspace_to_rgb(img)
        model(img.cuda())
        opt.zero_grad()
        loss = -1*hook_out[0][feature].mean()
        loss.backward()
        opt.step()
        if debug and (i)%(steps/frames)==0:
            clear_output(wait=True)
            show_rgb(image_buf_to_rgb(img_buf, jitter, **kwargs),
                     label=f"step: {i} loss: {loss}", **kwargs)

    hook.remove()
    
    retval = image_buf_to_rgb(img_buf, jitter, **kwargs)
    if show:
        if not debug: show_rgb(retval, **kwargs)
    else:
        return retval
