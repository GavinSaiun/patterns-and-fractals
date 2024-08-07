import torch
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
Y, X = np.mgrid[-1.8:1.8:0.005, -2.5:1.5:0.005]

# High-resolution zoom 
# Y, X = np.mgrid[-0.1:0.1:0.0003, -0.2:0.34:0.0005]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y) # important!

zs = z.clone() # Updated!
ns = torch.zeros_like(z)

# transfer to the GPU device
z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

# Burning Ship Fractal computation
for i in range(200):
    zs_real = torch.abs(torch.real(zs))
    zs_imag = torch.abs(torch.imag(zs))
    zs_ = torch.complex(zs_real, zs_imag)
    zs_ = zs_ * zs_ + z
    not_diverged = torch.abs(zs_) < 4.0
    ns += not_diverged
    zs = zs_

# plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,10))

def processFractal(a):

    """Display an array of iteration counts as a
    colorful picture of a fractal."""

    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))

    return a

# plotting
plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()
