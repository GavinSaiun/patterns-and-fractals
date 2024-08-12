import torch
import numpy as np
import matplotlib.pyplot as plt


print("PyTorch Version:", torch.__version__)

# ensures the right computational device is being used 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creates 2 multi-dimensional grids which are assigned variables
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# transfer to the GPU device
x = x.to(device)
y = y.to(device)

# Compute Gaussian
z = torch.exp(-(x**2+y**2)/2.0)

# Compute Sine
a = torch.sin(x) * torch.sin(y)

# Compute Cosine
b = torch.cos(x) * torch.cos(y)

# Multiply Gaussian and Sine
c = z * a

#plot
# print(z)
# print(x)
# plt.imshow(z.cpu().numpy())

plt.imshow(a.cpu().numpy())

# plt.imshow(c.cpu().numpy())

plt.tight_layout()
plt.show()

