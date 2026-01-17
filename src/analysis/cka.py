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

    def _centering(self, K):
        """Centers the kernel matrix K (HKH)."""
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def calculate(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Args:
            X: Embedding matrix A [N, D1]
            Y: Embedding matrix B [N, D2]
        """
        # Ensure centering of features first (optional but stable)
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        # Compute Gram Matrices
        # Note: For N > 20k, this might be memory intensive. 
        # We assume N = Test Set Size, which is usually manageable.
        gram_x = torch.matmul(X, X.T)
        gram_y = torch.matmul(Y, Y.T)

        # Center Gram Matrices
        gram_x_centered = self._centering(gram_x)
        gram_y_centered = self._centering(gram_y)

        # Compute HSIC (Hilbert-Schmidt Independence Criterion)
        # HSIC(K, L) = tr(K_centered @ L_centered)
        hsic_xy = torch.sum(gram_x_centered * gram_y_centered)
        hsic_xx = torch.sum(gram_x_centered * gram_x_centered)
        hsic_yy = torch.sum(gram_y_centered * gram_y_centered)

        cka_score = hsic_xy / (torch.sqrt(hsic_xx) * torch.sqrt(hsic_yy))
        
        return cka_score.item()