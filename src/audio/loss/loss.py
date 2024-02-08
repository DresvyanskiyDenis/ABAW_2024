import torch
import torch.nn.functional as F
import torch.nn as nn


class CCCLoss(nn.Module):
    """Lin's Concordance Correlation Coefficient: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Measures the agreement between two variables
    
    It is a product of
    - precision (pearson correlation coefficient) and
    - accuracy (closeness to 45 degree line)
    
    Interpretation
    - rho =  1: perfect agreement
    - rho =  0: no agreement
    - rho = -1: perfect disagreement
    
    Args:
        eps (float, optional): Avoiding division by zero. Defaults to 1e-8.
    """
    
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes CCC loss

        Args:
            x (Tensor): input tensor with shapes (n, 2); 0 - valence, 1 - arousal
            y (Tensor): target tensor with shapes (n, 2); 0 - valence, 1 - arousal

        Returns:
            Tensor: 1 - CCC loss value
        """
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + self.eps)
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1 - ccc


class VALoss(nn.Module):
    """Valence-Arousal loss of VA Estimation Challenge
    Computes weighted loss between valence and arousal CCC loses 
    
    Args:
        alpha (float, optional): Weighted coefficient for valence. Defaults to 1.
        beta (float, optional): Weighted coefficient for arousal. Defaults to 1.
        eps (float, optional): Avoiding division by zero. Defaults to 1e-8.
    """
    
    def __init__(self, alpha: float = 1, beta: float = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.ccc = CCCLoss(eps=eps)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes VA loss

        Args:
            x (Tensor): input tensor with shapes (n, 2); 0 - valence, 1 - arousal
            y (Tensor): target tensor with shapes (n, 2); 0 - valence, 1 - arousal

        Returns:
            Tensor: VA loss value
        """
        loss = self.alpha * self.ccc(x[:, 0], y[:, 0]) + self.beta * self.ccc(x[:, 1], y[:, 1])
        return loss