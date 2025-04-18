try:
    from sageattention import sageattn
except:
    sageattn = None

try:
    major, minor = torch.cuda.get_device_capability(0)
    if f"{major}.{minor}" == "8.0":
        from sageattention_sm80 import sageattn
    elif f"{major}.{minor}" == "8.6":
        from sageattention_sm86 import sageattn
    elif f"{major}.{minor}" == "8.9":
        from sageattention_sm89 import sageattn
    elif major>=9:
        from sageattention_sm90 import sageattn
except:
    try:
        from sageattention import sageattn
    except:
        sageattn = None

import torch
from yunchang.kernels import AttnType


def xdit_sage_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type=AttnType.FA,
    attn_processor=None,
    attn_layer=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if sageattn is not None:
        out = sageattn(q, k, v, is_causal=False)
    else:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).contiguous()
    return out