"""P4.2: muP, Depth-muP and CompleteP for the course language model (model and optimizer builders)."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from modeling import AutoregressiveLM, LlamaDecoderLayer, initialize_model
from optimizers import ADAMW_EPSILON, should_apply_weight_decay


DEPTH_RULES = ('none', 'depth-mup', 'completep')


def depth_multipliers(depth_rule, depth, ref_depth):
    """(residual multiplier c_L, block LR multiplier, block Adam eps multiplier) with r = depth / ref_depth."""
    r = depth / ref_depth
    return {'none': (1., 1., 1.), 'depth-mup': (r**-.5, r**-.5, r**-.5), 'completep': (1 / r, 1., 1 / r)}[depth_rule]


class ScaledDecoderLayer(LlamaDecoderLayer):
    """Course decoder layer with c_L multiplying the attention and feedforward branches."""
    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        residual = hidden_states
        branch = self.self_attn(self.input_layernorm(hidden_states), position_embeddings=position_embeddings,
                                attention_mask=attention_mask)
        hidden_states = residual + self.residual_multiplier * F.dropout(branch, p=self.dropout, training=self.training)
        residual = hidden_states
        branch = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + self.residual_multiplier * F.dropout(branch, p=self.dropout, training=self.training)


class ParameterizedLM(AutoregressiveLM):
    """Course model with the muP readout scaling and shared-Gaussian initialization of the 4.2 table.

    mup=False keeps the course baseline. With mup=True, m = width / ref_width: the embedding is G/n0 instead of
    G/n, the readout G/sqrt(n0) instead of G/sqrt(n), and the readout input is multiplied by 1/m. G is the same
    truncated Gaussian draw the baseline uses. The depth rule multiplies each residual branch by c_L.
    """
    def __init__(self, config, dtype=torch.float32, qk_norm=True, tie_word_embeddings=False, dropout=0.0,
                 *, mup=False, ref_width=512, ref_depth=8, depth_rule='none'):
        if tie_word_embeddings:
            raise ValueError('The 4.2 parameterization uses untied embedding and readout')
        if depth_rule not in DEPTH_RULES:
            raise ValueError(f'depth_rule must be one of {DEPTH_RULES}')
        super().__init__(config, dtype=dtype, qk_norm=qk_norm, tie_word_embeddings=False, dropout=dropout)
        self.mup, self.ref_width = mup, ref_width
        self.output_multiplier = ref_width / config.hidden_size if mup else 1.
        residual, _, _ = depth_multipliers(depth_rule, config.num_hidden_layers, ref_depth)
        if depth_rule != 'none':
            for layer in self.model.layers:
                layer.__class__ = ScaledDecoderLayer
                layer.residual_multiplier = residual

    def forward(self, input_ids=None, attention_mask=None, position_ids=None):
        if input_ids is None:
            raise ValueError('input_ids must be provided.')
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        return self.lm_head(hidden * self.output_multiplier)

    def initialize_parameters(self):
        initialize_model(self)
        if self.mup:
            m = self.config.hidden_size / self.ref_width
            with torch.no_grad():
                self.model.embed_tokens.weight.mul_(m)
                self.lm_head.weight.mul_(math.sqrt(m))


def build_optimizer(model, optimizer_name, learning_rate, weight_decay, beta1, beta2,
                    *, mup=False, ref_width=512, ref_depth=8, depth_rule='none'):
    """AdamW with the course's masked weight decay and per-group LR / eps.

    Embedding, readout and final norm use the base LR. Hidden block matrices use base LR / m under muP, and the
    depth rule scales block matrices and block norms (LR and eps). Weight decay stays at its configured value.
    """
    if optimizer_name != 'adamw':
        raise ValueError('The 4.2 builders support adamw')
    m = model.config.hidden_size / ref_width if mup else 1.
    _, block_lr, block_eps = depth_multipliers(depth_rule, model.config.num_hidden_layers, ref_depth)
    roles = {'embedding': (learning_rate, 1.), 'readout': (learning_rate, 1.), 'final_norm': (learning_rate, 1.),
             'block_norm': (learning_rate * block_lr, block_eps),
             'hidden': (learning_rate / m * block_lr, block_eps)}
    groups, seen = {}, set()
    for module_name, module in model.named_modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            in_block = '.layers.' in f'.{module_name}.'
            if in_block:
                role = 'hidden' if isinstance(module, nn.Linear) else 'block_norm'
            elif isinstance(module, nn.Embedding):
                role = 'embedding'
            elif isinstance(module, nn.Linear):
                role = 'readout'
            else:
                role = 'final_norm'
            decay = weight_decay if should_apply_weight_decay(module, name) else 0.
            groups.setdefault((role, decay), []).append(p)
    param_groups = [dict(params=params, lr=roles[role][0], eps=ADAMW_EPSILON * roles[role][1], weight_decay=decay)
                    for (role, decay), params in groups.items()]
    kwargs = dict(lr=learning_rate, betas=(beta1, beta2), eps=ADAMW_EPSILON, weight_decay=0.)
    if any(p.is_cuda for g in param_groups for p in g['params']):
        kwargs['fused'] = True
    return torch.optim.AdamW(param_groups, **kwargs)
