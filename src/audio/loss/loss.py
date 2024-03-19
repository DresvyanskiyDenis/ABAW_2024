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
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: 1 - CCC loss value
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
            x (Tensor): Input tensor with shapes (n, 2); 0 - valence, 1 - arousal
            y (Tensor): Target tensor with shapes (n, 2); 0 - valence, 1 - arousal

        Returns:
            torch.Tensor: VA loss value
        """
        loss = self.alpha * self.ccc(x[:, 0], y[:, 0]) + self.beta * self.ccc(x[:, 1], y[:, 1])
        return loss


class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance. 
    Implemented by ***. # TODO
    
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    
    Args:
        alpha (torch.Tensor, optional): Weights for each class. Defaults to None.
        gamma (float, optional): A constant, as described in the paper. Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'. Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore. Defaults to -100.
        
    Raises:
        ValueError: Supported reduction types: "mean", "sum", "none"
    """

    def __init__(self,
                 alpha: torch.Tensor = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100) -> None:
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self) -> str:
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes Focal Loss

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: Focal Loss value
        """
        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class SoftFocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002 with soft targets, 
    For example, target can be [0, 0.3, 0.7, 1]
    Class FocalLoss takes only digit targets. 
    Implemented by ***. # TODO
    
    Args:
        softmax (bool): Apply softmax or not. Defaults to True.
        alpha (torch.Tensor, optional): Weights for each class. Defaults to None.
        gamma (float, optional): A constant, as described in the paper. Defaults to 0.
    """
    def __init__(self,
                 softmax: bool = True,
                 alpha: torch.Tensor = None,
                 gamma: float = 0.) -> None:
        super().__init__()
        self.alpha = 1 if alpha is None else alpha
        self.gamma = gamma
        self.softmax = softmax

    def __repr__(self) -> str:
        arg_keys = ['alpha', 'gamma']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes Focal Loss for soft targets

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: Focal Loss value
        """
        if self.softmax:
            p = F.softmax(x, dim=-1)
        else:
            p = x

        epsilon = 1e-7
        p = torch.clip(p, epsilon, 1. - epsilon)
        cross_entropy = -y * torch.log(p)

        # focal loss
        loss = self.alpha * torch.pow(1. - p, self.gamma) * cross_entropy
        loss = torch.sum(loss, dim =-1).mean()
        return loss


class SoftFocalLossWrapper(nn.Module):
    """Wrapper for FocalLoss class
    Performs one-hot encoding

    Args:
        focal_loss (nn.Module): Focal loss
        num_classes (int): Number of classes
    """
    def __init__(self,
                 focal_loss: nn.Module,
                 num_classes: int) -> None:

        super().__init__()
        self.focal_loss = focal_loss
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes Focal Loss for soft targets

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: Focal Loss value
        """
        new_y = F.one_hot(y, num_classes=self.num_classes)
        return self.focal_loss(x, new_y)


class VAELoss(nn.Module):
    """Weighted sum of VA and Expr losses

    Args:
        va_loss (nn.Module): Valence/arousal loss function
        expr_loss (nn.Module): Expression loss function
        alpha (float, optional): Weighted coefficient. Defaults to 0.5.
    """
    def __init__(self,
                 va_loss: nn.Module,
                 expr_loss: nn.Module,
                 alpha: float = .5) -> None:
        super().__init__()
        self.va_loss = va_loss
        self.expr_loss = expr_loss
        self.alpha = alpha

    def forward(self, x: list[torch.Tensor], y: list[torch.Tensor]) -> torch.Tensor:
        """Computes joint va/expr loss
        Flatten expression labels before loss calculation

        Args:
            x (list[torch.Tensor]): Input list of tensors; 0 - va, 1 - expr
            y (list[torch.Tensor]): Target list of tensors; 0 - va, 1 - expr

        Returns:
            torch.Tensor: Weighted loss value
        """
        num_classes = x[1].shape[-1]
        return self.alpha * self.va_loss(x[0], y[0]) + (1 - self.alpha) * self.expr_loss(x[1].reshape(-1, num_classes), y[1].flatten())


if __name__ == "__main__":
    num_classes = 8
    batch_size = 6
    va_x = torch.randn(batch_size, 20, 2)
    expr_x = torch.randn(batch_size, 4, num_classes)

    va_y = torch.randn(batch_size, 20, 2)
    expr_y = torch.rand(batch_size, 4, num_classes)
    expr_y = expr_y.reshape(-1, num_classes)
    expr_y = torch.argmax(expr_y, axis=1)
    expr_y = expr_y.reshape(batch_size, 4)

    expr_loss = SoftFocalLossWrapper(focal_loss=SoftFocalLoss(alpha=None), num_classes=num_classes)
    va_loss = VALoss()
    vae_loss = VAELoss(va_loss=va_loss, expr_loss=expr_loss)

    print(vae_loss([va_x, expr_x], [va_y, expr_y]))
