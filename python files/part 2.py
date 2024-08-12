import torch
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]


# High-resolution zoom 
Y, X = np.mgrid[-0.9:0.11:0.001, -0.7:0.7:0.001]

# load into PyTorch tensors, converts numpy array into a tensor
x = torch.Tensor(X)
y = torch.Tensor(Y)

# combines x, y into a single tensor of form z = x + iy
z = torch.complex(x, y) #important!

zs = z.clone() #Updated!

# creates array filled with 0's with the same shape as 'z'
ns = torch.zeros_like(z)

# transfer to the GPU device
z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

#Mandelbrot Set
# a collection of complex numbers that do not diverge under 
# repeated iterations
# z0 is fixed, c is not constant, identifies which c value generates a bounded sequence
for i in range(200):
    #Compute the new values of z: z^2 + x
    zs_ = zs*zs + z
    #Have we diverged with this new value?
    not_diverged = torch.abs(zs_) < 4.0
    #Update variables to compute
    ns += not_diverged
    zs = zs_

# Julia Set computation
# Definition: a collection of complex numbers that do not diverge under
# repeated iterations
# z is not fixed, c is a constant, identifies which initial value of z generates a bounded sequence
# c = torch.complex(torch.tensor(-0.7), torch.tensor(0.3)).to(device)
# for i in range(200):
#     # Julia Set Formula
#     zs_ = zs * zs + c
#     # Checks what elements have not diverged i.e. less than 4
#     not_diverged = torch.abs(zs_) < 4.0

#     # Count how many points not diverged
#     ns += not_diverged

#     # Update for next iteration
#     zs = zs_


fig = plt.figure(figsize=(16,10))

def processFractal(a):

    """Display an array of iteration counts as a
    colorful picture of a fractal."""

    # Reshape array for color processing 6.28 = 2pi
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])

    # combines RGB to create RGB colours
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)

    # Set colour of points that have not diverged to black
    img[a==a.max()] = 0
    a = img

    a = np.uint8(np.clip(a, 0, 255))

    return a

# plotting
plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()


