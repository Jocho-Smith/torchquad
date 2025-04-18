import torch
import torch.nn as nn
import math
from itertools import product


def compositions(n, d):
    """
    Generate all d-dimensional multi-indices (tuples) of non-negative integers summing to n.
    """
    if d == 1:
        yield (n,)
    else:
        for i in range(n + 1):
            for tail in compositions(n - i, d - 1):
                yield (i,) + tail


class SparseGridLayer(nn.Module):
    """
    PyTorch layer implementing sparse grid interpolation via combination technique.
    """
    def __init__(self, input_dim, level, boundary=True, domain=None):
        super().__init__()
        self.d = input_dim
        self.level = level
        self.boundary = boundary
        self.domain = domain or [(0.0, 1.0)] * self.d

        # Build subgrid nodes and combination coefficients
        l_vecs, i_vecs, coeffs = self._generate_sparse_basis()
        self.n_basis = len(coeffs)
        self.register_buffer('levels', torch.tensor(l_vecs, dtype=torch.long))  # (n_basis,d)
        self.register_buffer('indices', torch.tensor(i_vecs, dtype=torch.long)) # (n_basis,d)
        self.register_buffer('coeffs', torch.tensor(coeffs, dtype=torch.float)) # (n_basis,)

    def _generate_sparse_basis(self):
        l_vectors, i_vectors, coeffs = [], [], []
        l_min, l_max = self.level, self.level + self.d - 1
        for s in range(l_min, l_max + 1):
            for l_vec in compositions(s, self.d):
                if any(li < 1 for li in l_vec):
                    continue
                k = l_max - s
                coeff = ((-1) ** k) * math.comb(self.d - 1, k)
                # full-grid nodal indices
                idx_lists = []
                for li in l_vec:
                    max_i = 2 ** li
                    if self.boundary:
                        raw = list(range(0, max_i + 1))
                    else:
                        raw = list(range(1, max_i))
                    idx_lists.append(raw)
                for i_vec in product(*idx_lists):
                    l_vectors.append(l_vec)
                    i_vectors.append(i_vec)
                    coeffs.append(coeff)
        return l_vectors, i_vectors, coeffs

    def compute_phi(self, x):
        """Evaluate full-grid nodal hat basis functions for all subgrid nodes."""
        # normalize inputs
        lows = torch.tensor([l for l, _ in self.domain], device=x.device)
        highs = torch.tensor([h for _, h in self.domain], device=x.device)
        x_norm = (x - lows) / (highs - lows)
        B = x.shape[0]
        # broadcast and compute hat
        x_exp = x_norm.unsqueeze(1)                   # (B,1,d)
        pow2 = 2 ** self.levels.float()               # (n_basis,d)
        dist = torch.abs(x_exp * pow2.unsqueeze(0) - self.indices.unsqueeze(0))
        phi = torch.clamp(1 - dist, min=0)            # (B,n_basis,d)
        return torch.prod(phi, dim=2)                 # (B,n_basis)

    def forward(self, x, f=None):
        """Return sparse grid interpolation of f on nodes if f given, else weighted sum using registered weights."""
        if f is not None:
            # direct interpolation: compute nodal values and return u(x)
            # compute node coordinates
            levels = self.levels.float()
            indices = self.indices.float()
            x_nodes = indices / (2 ** levels)
            # map to domain
            lows = torch.tensor([l for l, _ in self.domain], device=x.device)
            highs = torch.tensor([h for _, h in self.domain], device=x.device)
            x_nodes = lows + x_nodes * (highs - lows)
            # evaluate f at nodes
            y_nodes = f(x_nodes)
            # weights = y_nodes
            weights = y_nodes
            # combine
            phi = self.compute_phi(x)
            eff = self.coeffs.unsqueeze(0) * weights.unsqueeze(0)
            return torch.sum(phi * eff, dim=1)
        else:
            # assume weights are assigned to self.weights
            phi = self.compute_phi(x)
            eff = self.coeffs.unsqueeze(0) * self.weights.unsqueeze(0)
            return torch.sum(phi * eff, dim=1)

# Example unit tests with collocation fit

def _test_function_approx(level, dim, func, tol=1e-2):
    """Test interpolation accuracy on training data."""
    layer = SparseGridLayer(dim, level)
    N = 200
    x = torch.rand(N, dim)
    # direct interpolation
    y_pred = layer(x, f=func)
    y_true = func(x)
    mse = torch.mean((y_pred - y_true) ** 2)
    assert mse < tol, f"MSE {mse.item()} > tol {tol}"


def test_all():
    funcs = [
        lambda x: torch.sum(x ** 2, dim=1),
        lambda x: torch.sum(torch.exp(x), dim=1),
        lambda x: torch.sum(torch.sin(x), dim=1),
    ]
    for d in [1, 2, 3]:
        for f in funcs:
            _test_function_approx(level=3, dim=d, func=f)
    print("All sparse grid tests passed.")

if __name__ == '__main__':
    test_all()
