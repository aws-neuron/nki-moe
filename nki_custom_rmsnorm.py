import torch
import torch.nn as nn
import nki
import nki.isa as nisa
import nki.language as nl


def stream_shuffle_broadcast(src, dst):
    """Broadcast a single-partition tile to all partitions of dst."""
    dst_npar = dst.shape[0]
    free_dim = dst.shape[1]
    shuffle_mask = [0] * 32

    assert dst_npar % 32 == 0
    for i in range(dst_npar // 32):
        nisa.nc_stream_shuffle(
            src=src[0:1, :],
            dst=dst[i * 32 : (i + 1) * 32, 0:free_dim],
            shuffle_mask=shuffle_mask,
        )


@nki.jit()
def nki_rmsnorm_kernel(input_tensor, weight, eps):
    """
    RMSNorm NKI kernel - migrated to NKI Beta 2 API.

    Args:
        input_tensor: Input tensor [batch*seq_len, hidden_size]
        weight: RMSNorm weight parameter [hidden_size]
        eps: Small epsilon for numerical stability

    Returns:
        output: Normalized tensor with same shape as input
    """
    MAX_P = 128

    output = nl.ndarray(
        input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm
    )
    assert input_tensor.shape[1] == weight.shape[0]

    num_rows = input_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    num_chunks = (num_rows + MAX_P - 1) // MAX_P

    # Load RMSNorm weight once into SBUF, reused by all rows
    g_tile = nl.ndarray((1, hidden_size), dtype=weight.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=g_tile[0:1, 0:hidden_size],
        src=weight.reshape((1, hidden_size))[0:1, 0:hidden_size],
    )

    for i in nl.affine_range(num_chunks):
        p_start = i * MAX_P
        valid_rows = min(MAX_P, num_rows - p_start)

        # Load valid rows from HBM
        a = nl.ndarray((MAX_P, hidden_size), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=a[0:valid_rows, 0:hidden_size],
            src=input_tensor[p_start : p_start + valid_rows, 0:hidden_size],
        )

        # a^2 -> t
        t = nl.ndarray((MAX_P, hidden_size), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t, data1=a, data2=a, op=nl.multiply)

        # sum(a^2)
        sq_sum = nl.ndarray((MAX_P, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_reduce(dst=sq_sum, data=t, op=nl.add, axis=1)

        # rsqrt(mean(a^2) + eps), in-place
        s = nl.ndarray((MAX_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=s,
            data=sq_sum,
            op0=nl.multiply,
            operand0=1.0 / hidden_size,
            op1=nl.add,
            operand1=eps,
        )
        nisa.activation(dst=s, data=s, op=nl.rsqrt)

        # a * rsqrt -> t
        nisa.tensor_scalar(dst=t, data=a, operand0=s, op0=nl.multiply)

        # Broadcast weight and multiply
        g_bcast = nl.ndarray((MAX_P, hidden_size), dtype=g_tile.dtype, buffer=nl.sbuf)
        stream_shuffle_broadcast(g_tile, g_bcast)
        nisa.tensor_tensor(dst=t, data1=t, data2=g_bcast, op=nl.multiply)

        # Store only valid rows back to HBM
        nisa.dma_copy(
            dst=output[p_start : p_start + valid_rows, 0:hidden_size],
            src=t[0:valid_rows, 0:hidden_size],
        )

    return output


class NKIRMSNorm(nn.Module):
    """
    NKI-accelerated RMSNorm layer compatible with NxDI.
    """
    
    def __init__(self, hidden_size, eps=1e-6):
        """
        Initialize NKI RMSNorm layer.
        
        Args:
            hidden_size: Size of the hidden dimension
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
        # Enable NKI kernel for all hidden sizes
        self.use_nki = True
        
        print(f"Info: Using NKI RMSNorm kernel for hidden_size={hidden_size}")
        
    def forward(self, x):
        """
        Forward pass using NKI kernel or fallback.
        
        Args:
            x: Input tensor of various shapes
            
        Returns:
            Normalized tensor with same shape as input
        """
        if not self.use_nki:
            return self._fallback_forward(x)
        
        original_shape = x.shape
        
        # Handle various input shapes by flattening to 2D
        if x.dim() >= 2:
            x = x.view(-1, x.shape[-1])
        else:
            raise ValueError(f"Expected input with at least 2 dimensions, got {x.dim()}D")
        
        # Call NKI kernel directly
        # .view(-1) materializes PlaceholderParameter into a real torch.Tensor
        # during torch_neuronx tracing, so NKI's tracer can introspect it
        output = nki_rmsnorm_kernel(x, self.weight.view(-1), self.eps)
        
        # Reshape back to original shape
        output = output.view(original_shape)
        
        return output
    
    def _fallback_forward(self, x):
        """Fallback RMSNorm implementation."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x / torch.sqrt(variance + self.eps)
        return x_normalized * self.weight
    
    def extra_repr(self):
        """String representation for debugging."""
        return f'hidden_size={self.hidden_size}, eps={self.eps}, use_nki={self.use_nki}'


def get_nki_rmsnorm_cls():
    """
    Factory function to return NKI RMSNorm class.
    """
    return NKIRMSNorm


def numpy_rmsnorm(x, weight, eps):
    """Reference RMSNorm in pure NumPy."""
    import numpy as np
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    x_norm = x / np.sqrt(variance + eps)
    return x_norm * weight


if __name__ == "__main__":
    import numpy as np
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()

    hidden_size = 512
    eps = 1e-6

    np.random.seed(42)
    x_np = np.random.randn(256, hidden_size).astype(np.float32)
    w_np = np.random.randn(hidden_size).astype(np.float32)

    # NumPy golden output
    expected = numpy_rmsnorm(x_np, w_np, eps)

    # NKI kernel output
    x_torch = torch.from_numpy(x_np).to(device)
    w_torch = torch.from_numpy(w_np).to(device)
    nki_out = nki_rmsnorm_kernel(x_torch, w_torch, eps)
    actual = nki_out.cpu().numpy()

    max_diff = np.max(np.abs(expected - actual))
    mean_diff = np.mean(np.abs(expected - actual))

    print(f"Shape:     {x_np.shape}")
    print(f"Max diff:  {max_diff:.6e}")
    print(f"Mean diff: {mean_diff:.6e}")

    if max_diff < 1e-3:
        print("PASS")
    else:
        print("FAIL")
