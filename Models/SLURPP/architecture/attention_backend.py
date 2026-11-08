"""Use xFormers attention when installed, with equivalent PyTorch scaled dot-product attention as a fallback."""
import torch.nn.functional as F

try:
    from xformers.ops import memory_efficient_attention as _attention
except ImportError:
    _attention = None


def memory_efficient_attention(query, key, value, scale=None):
    """Attention inputs and scale -> attention result with unchanged dimensions."""
    if _attention is not None:
        return _attention(query, key, value, scale=scale)
    return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, scale=scale)
