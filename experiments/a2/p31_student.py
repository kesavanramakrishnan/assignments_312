"""Student entry point for P4.1. Complete the two functions below."""
import argparse
from experiments.a2.stress import StressConfig, load_tokens, run


POLICIES = ('kaiming', 'mup', 'depth-mup', 'completep')
BASE_EPS = 1e-8
HIDDEN = ('q', 'k', 'v', 'o', 'gate', 'up', 'down')
BLOCK_NORMS = ('norm1', 'norm2', 'qnorm', 'knorm')
# Set by main(): the prescription under test and its reference width/depth.
POLICY = dict(name='kaiming', ref_width=512, ref_depth=2)


def _is_mup():
    return POLICY['name'] != 'kaiming'


def _depth_rules(depth):
    """(residual multiplier c_L, block LR multiplier, block Adam eps multiplier) with r = L / reference depth."""
    r = depth / POLICY['ref_depth']
    return {'kaiming': (1., 1., 1.), 'mup': (1., 1., 1.), 'depth-mup': (r**-.5, r**-.5, r**-.5),
            'completep': (1 / r, 1., 1 / r)}[POLICY['name']]


def initialize(model):
    """Independent centered Gaussians: embedding variance 1, hidden variance 1/fan-in, norm gains 1,
    readout variance 1/n (Kaiming) or 1/n0 (muP). Forward multipliers come from the selected policy."""
    n, n0 = model.config.width, POLICY['ref_width']
    for name, p in model.named_parameters():
        leaf = name.split('.')[-2]
        if leaf == 'embed':
            p.normal_(0., 1.)
        elif leaf == 'head':
            p.normal_(0., ((n0 if _is_mup() else n) ** -1) ** .5)
        elif leaf in BLOCK_NORMS or leaf == 'norm':
            p.fill_(1.)
        elif leaf in HIDDEN:
            p.normal_(0., (1 / p.shape[1]) ** .5)
        else:
            raise ValueError(f'unhandled parameter {name}')
    model.output_multiplier = n0 / n if _is_mup() else 1.
    residual, _, _ = _depth_rules(len(model.blocks))
    for block in model.blocks:
        block.residual_multiplier = residual


def parameter_groups(model, base_lr):
    """Adam groups: embedding/readout/final norm at the base LR; hidden matrices at base LR / m under muP;
    depth prescriptions scale the block matrices and block norms (LR and eps) by their block multipliers."""
    m = model.config.width / POLICY['ref_width'] if _is_mup() else 1.
    _, block_lr, block_eps = _depth_rules(len(model.blocks))
    buckets = dict(outer=[], block_norm=[], hidden=[])
    for name, p in model.named_parameters():
        leaf = name.split('.')[-2]
        if leaf in HIDDEN:
            buckets['hidden'].append(p)
        elif leaf in BLOCK_NORMS:
            buckets['block_norm'].append(p)
        else:
            buckets['outer'].append(p)
    return [dict(params=buckets['outer'], lr=base_lr, eps=BASE_EPS),
            dict(params=buckets['block_norm'], lr=base_lr * block_lr, eps=BASE_EPS * block_eps),
            dict(params=buckets['hidden'], lr=base_lr / m * block_lr, eps=BASE_EPS * block_eps)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-path', required=True)
    p.add_argument('--val-path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--head-dim', type=int, default=64)
    p.add_argument('--precision', choices=('fp32', 'mp'), default='fp32',
                   help='fp32 for width comparisons; mp for depth comparisons')
    p.add_argument('--microbatch', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-alignment', action='store_true')
    p.add_argument('--policy', choices=POLICIES, default='kaiming',
                   help='kaiming and mup are the width prescriptions; depth-mup and completep add depth rules')
    p.add_argument('--ref-width', type=int, default=512, help='muP reference width n0 (64 for the depth runs)')
    p.add_argument('--ref-depth', type=int, default=2)
    p.add_argument('--no-wandb', action='store_true', help='Save diagnostics locally without W&B')
    a = p.parse_args(argv)
    c = StressConfig(width=a.width, depth=a.depth, head_dim=a.head_dim,
                     microbatch=a.microbatch, seed=a.seed, precision=a.precision)
    c.validate()
    POLICY.update(name=a.policy, ref_width=a.ref_width, ref_depth=a.ref_depth)
    train = load_tokens(a.train_path, c.steps * c.batch, c.context)
    val = load_tokens(a.val_path, c.batch, c.context)
    result = run(c, train, val, base_lr=a.lr, initialize_fn=initialize,
                 groups_fn=parameter_groups, device=a.device,
                 alignment=not a.no_alignment, output=a.output)
    if not a.no_wandb:
        import wandb
        from pathlib import Path
        from utils import WANDB_ENTITY, WANDB_PROJECT
        from experiments.a2.wandb_diagnostics import log_records
        with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
                        name=Path(a.output).stem, tags=['a2', 'p4.1', f'p4.1-{a.policy}'],
                        config={**result['config'], 'base_lr': a.lr, 'policy': a.policy,
                                'ref_width': a.ref_width, 'ref_depth': a.ref_depth,
                                'parameter_groups': result['parameter_groups'],
                                'data_sha256': result['data_sha256']}) as wb:
            wandb.define_metric('optimizer_step')
            wandb.define_metric('*', step_metric='optimizer_step')
            for row in result['history']:
                wandb.log({'optimizer_step': row['step'],
                           **{key: row[key] for key in ('val_loss', 'train_loss', 'logit_rms') if key in row}})
            log_records(wandb, result['alignment'], result['history'])
            wandb.save(a.output, base_path=str(Path(a.output).parent), policy='now')
            print('WANDB_RUN_URL=' + wb.url)
    print('Final validation loss:', result['history'][-1]['val_loss'])


if __name__ == '__main__':
    main()
