import torch

class LinearCKA:
    """
    Implements Linear Centered Kernel Alignment (CKA).
    
    Why CKA?
    Standard metrics (MSE, Cosine) fail if two models learn the same geometry 
    but with different rotations/permutations of dimensions. 
    CKA is invariant to orthogonal transformations, making it ideal for 
    comparing Deep Learning representations trained with different seeds/matrices.
    
    Formula:
        CKA(X, Y) = <vec(XX'), vec(YY')> / (||XX'||_F * ||YY'||_F)
    """
    
    def __init__(self, device=None):
        self.device = device or torch.device('cpu')

    def calculate(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Args:
            X: Embedding matrix A [N, D1]
            Y: Embedding matrix B [N, D2]

        Uses the feature-space form of linear CKA:

            HSIC(X, Y) = || X^T Y ||_F^2       (for mean-centered X, Y)
            CKA(X, Y)  = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))

        Numerically identical to the Gram-matrix form HSIC(K_X, K_Y) =
        tr(K_X_centered K_Y_centered), but O(N*D^2) instead of O(N^3) and
        never materializes an N x N matrix.  For N=12k, D=64 this is a
        ~1000x speed up and a ~1000x memory reduction vs. the n-by-n form.
        """
        X = X.to(self.device) - X.mean(dim=0, keepdim=True)
        Y = Y.to(self.device) - Y.mean(dim=0, keepdim=True)

        hsic_xy = (X.T @ Y).pow(2).sum()
        hsic_xx = (X.T @ X).pow(2).sum()
        hsic_yy = (Y.T @ Y).pow(2).sum()

        cka_score = hsic_xy / (torch.sqrt(hsic_xx) * torch.sqrt(hsic_yy))
        return cka_score.item()