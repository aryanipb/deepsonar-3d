"""AdaPoinTr (Yu et al., TPAMI 2023) ported from https://github.com/yuxumin/PoinTr (MIT)."""
import torch
from easydict import EasyDict

from .AdaPoinTr import AdaPoinTr

# cfgs/PCN_models/AdaPoinTr.yaml from the official repo
PCN_CONFIG = EasyDict({
    'NAME': 'AdaPoinTr', 'num_query': 512, 'num_points': 16384, 'center_num': [512, 256],
    'global_feature_dim': 1024, 'encoder_type': 'graph', 'decoder_type': 'fc',
    'encoder_config': {
        'embed_dim': 384, 'depth': 6, 'num_heads': 6, 'k': 8, 'n_group': 2, 'mlp_ratio': 2.,
        'block_style_list': ['attn-graph', 'attn', 'attn', 'attn', 'attn', 'attn'], 'combine_style': 'concat',
    },
    'decoder_config': {
        'embed_dim': 384, 'depth': 8, 'num_heads': 6, 'k': 8, 'n_group': 2, 'mlp_ratio': 2.,
        'self_attn_block_style_list': ['attn-graph'] + ['attn'] * 7, 'self_attn_combine_style': 'concat',
        'cross_attn_block_style_list': ['attn-graph'] + ['attn'] * 7, 'cross_attn_combine_style': 'concat',
    },
})


def load_adapointr(ckpt_path, device='cpu'):
    model = AdaPoinTr(PCN_CONFIG)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = state.get('base_model', state.get('model', state))
    sd = {k.removeprefix('module.'): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()
