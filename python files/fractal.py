import torch
import matplotlib.pyplot as plt


"""
My implementation of a fractal, named Ikeda Map
the Ikeda map is a discrete-time dynamical system given by the formula
    x1 = 1 + u * (x * cos(t) - y * sin(t))
    y1 = u * (x * sin(t) + y * cos(t))
    t = 0.4 - 6 / (1 + x ** 2 + y ** 2)
For u ≥ 0.6, this system has a chaotic attractor.
"""
def ikeda_map(u: float, points=100, iterations=300):
    """
    Args:
        u:float
            ikeda parameter
        points:int
            number of starting points
        iterations:int
            number of iterations
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x = 10 * torch.randn(points, 1, device=device)
    y = 10 * torch.randn(points, 1, device=device)
    
    for n in range(points):
        X = compute_ikeda_trajectory(u, x[n][0], y[n][0], iterations, device)

        plot_ikeda_trajectory(X)
    
    return plt

def compute_ikeda_trajectory(u: float, x: float, y: float, N: int, device: torch.device):
    """Calculate a full trajectory

    Args:
        u - is the ikeda parameter
        x, y - coordinates of the starting point
        N - the number of iterations
        device - the device to perform computations on

    Returns:
        A PyTorch tensor of shape (N, 2).
    """
    X = torch.zeros((N, 2), device=device)
    
    for n in range(N):
        X[n] = torch.tensor([x, y], device=device)
        
        t = 0.4 - 6 / (1 + x ** 2 + y ** 2)
        x1 = 1 + u * (x * torch.cos(t) - y * torch.sin(t))
        y1 = u * (x * torch.sin(t) + y * torch.cos(t))
        
        x = x1
        y = y1   
        
    return X

def plot_ikeda_trajectory(X) -> None:
    """
    Plot the whole trajectory

    Args:
        X: torch.Tensor
            trajectory of an associated starting-point
    """
    plt.plot(X[:,0].cpu().numpy(), X[:, 1].cpu().numpy(), "k")

# Plot the Ikeda Map
ikeda_map(0.918).show()

