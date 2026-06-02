import torch, os, sys

sd = torch.load(os.path.expandvars(
    r'%USERPROFILE%\AppData\Roaming\hearth\models\Bonsai-Image-Ternary-4B-Gemlite\transformer-gemlite-int2\state_dict.pt'
), map_location='cpu', weights_only=True)

names = [
    'single_transformer_blocks.0.attn.to_out',
    'single_transformer_blocks.0.attn.to_qkv_mlp_proj',
    'transformer_blocks.0.attn.to_q',
    'transformer_blocks.0.ff.linear_in',
    'transformer_blocks.0.ff_context.linear_in',
]

for name in names:
    print(f"\n{name}:")
    for suffix in ['W_q', 'scales', 'zeros', 'metadata', 'orig_shape', 'bias']:
        key = f'{name}.{suffix}'
        if key in sd:
            t = sd[key]
            if t.numel() <= 20:
                print(f"  {suffix}: shape={list(t.shape)} dtype={t.dtype} values={t.numpy().tolist()}")
            else:
                print(f"  {suffix}: shape={list(t.shape)} dtype={t.dtype}")

# Check if there are .weight keys too
weight_keys = [k for k in sd if k.endswith('.weight')]
print(f"\n.weight keys: {len(weight_keys)}")
for k in sorted(weight_keys)[:10]:
    print(f"  {k}: {list(sd[k].shape)}")
