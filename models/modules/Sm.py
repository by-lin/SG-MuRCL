import torch


class ApproxSm(torch.nn.Module):
    def __init__(self, alpha=0.5, num_steps=1) -> None:
        super().__init__()
        self.alpha = alpha
        self.num_steps = num_steps

        if isinstance(self.alpha, float):
            self.coef = (1.0/(1.0-self.alpha)-1)
        elif self.alpha == 'trainable':
            self.coef = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else:
            raise ValueError("alpha must be float or 'trainable'")
            
    def forward(self, f, A_mat):
        """
        input:
            f: tensor (batch_size, bag_size, d_dim)
            A_mat: sparse coo tensor (batch_size, bag_size, bag_size)
        output:
            g: tensor (batch_size, bag_size, d_dim)
        """

        # torch.sparse bug: this is a workaround
        recover_f = False
        if f.shape[2] == 1:
            recover_f = True
            f = torch.stack([f, f], dim=2).squeeze(-1) # (batch_size, bag_size, 2)

        g = f
        alpha = 1.0 / (1.0 + self.coef)
        for _ in range(self.num_steps):
            g = (1.0 - alpha)*f + alpha*torch.bmm(A_mat, g) # (batch_size, bag_size, d_dim)

        if recover_f:
            g = g[:, :, 0].unsqueeze(-1) # (batch_size, bag_size, 1)
        
        return g

class ExactSm(torch.nn.Module):
    def __init__(self, alpha=0.5) -> None:
        super().__init__()
        self.alpha = alpha
        
        if isinstance(self.alpha, float):
            self.coef = (1.0/(1.0-self.alpha)-1)
        elif self.alpha == 'trainable':
            self.coef = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else:
            raise ValueError("alpha must be float or 'trainable'")

    def forward(self, f, A_mat):
        """
        input:
            f: tensor (batch_size, bag_size, d_dim)
            lap_mat: sparse coo tensor (batch_size, bag_size, bag_size)
        output:
            g: tensor (batch_size, bag_size, d_dim)
        """
        batch_size = f.shape[0]
        bag_size = f.shape[1]

        id_mat = torch.eye(bag_size, device=A_mat.device).unsqueeze(0).repeat(batch_size, 1, 1) # (batch_size, bag_size, bag_size)

        M = (1+self.coef)*id_mat - self.coef*A_mat # (batch_size, bag_size, bag_size)
        g = self._solve_system(M, f) # (batch_size, bag_size, d_dim)
        return g

    def _solve_system(self, A, b):
        """
        input:
            A: tensor (batch_size, bag_size, bag_size)
            b: tensor (batch_size, bag_size, dim)
        output:
            x: tensor (batch_size, bag_size, d_dim)
        """
        x = torch.linalg.solve(A, b)
        return x
