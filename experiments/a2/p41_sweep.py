"""P4.1: launch five-step LR sweeps in parallel on Modal (detached; JSON results land in the private volume)."""
import argparse
import math

from data import DEFAULT_DATASET_DIR_NAME
from modal_utils import (app, build_image, user_volume, VOLUME_MOUNTS, MODAL_ENVIRONMENT,
                         MODAL_SHARED_DATASETS_DIR, timestamped_modal_app_name, secrets)


MAX_PARALLEL_RUNS = 10
WIDTH_POLICIES = ('kaiming', 'mup')
DEPTH_POLICIES = ('mup', 'depth-mup', 'completep')
REFERENCE_WIDTH = 512
DEPTH_WIDTH = 64
PILOT_LRS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100', retries=0, timeout=2 * 3600)
def _run(arguments):
    from experiments.a2.p31_student import main
    try:
        main(arguments)
    finally:
        user_volume.commit()


def width_job(policy, width, lr):
    name = f'p41-width-{policy}-w{width}-lr{lr:g}.json'
    return dict(name=name, args=['--policy', policy, '--ref-width', str(REFERENCE_WIDTH), '--width', str(width),
                                 '--depth', '2', '--head-dim', '64', '--precision', 'fp32', '--lr', repr(lr)])


def depth_job(policy, depth, lr):
    name = f'p41-depth-{policy}-d{depth}-lr{lr:g}.json'
    return dict(name=name, args=['--policy', policy, '--ref-width', str(DEPTH_WIDTH), '--ref-depth', '2',
                                 '--width', str(DEPTH_WIDTH), '--depth', str(depth), '--head-dim', '64',
                                 '--precision', 'mp', '--lr', repr(lr), '--no-alignment'])


def pilot():
    """Coarse LR grids at the smallest width (both prescriptions) and at depth 2, where all depth rules coincide."""
    jobs = [width_job(policy, 640, lr) for policy in WIDTH_POLICIES for lr in PILOT_LRS]
    return jobs + [depth_job('mup', 2, lr) for lr in PILOT_LRS]


# Grids centered on the pilot: width-640 optimum near 1e-3 for both prescriptions, depth-2 optimum near 1-3e-2.
# Kaiming grids step down by the expected 1/width shrinkage; muP grids stay around the width-640 optimum.
WIDTH_LRS = {
    ('mup', 640): (5e-4, 2e-3), ('kaiming', 640): (5e-4, 2e-3),
    ('mup', 2560): (2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3), ('kaiming', 2560): (6.25e-5, 1.25e-4, 2.5e-4, 5e-4, 1e-3),
    ('mup', 5120): (2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3), ('kaiming', 5120): (3.125e-5, 6.25e-5, 1.25e-4, 2.5e-4, 5e-4),
}
# Standard muP is expected to need a smaller LR as depth grows; the depth-aware rules should hold near 1e-2.
DEPTH_LRS = {
    ('mup', 2): (5e-3, 2e-2),
    ('mup', 100): (1e-4, 3e-4, 1e-3, 3e-3, 1e-2), ('mup', 1000): (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3),
    ('depth-mup', 100): (5e-3, 1e-2, 2e-2, 4e-2), ('depth-mup', 1000): (5e-3, 1e-2, 2e-2, 4e-2),
    ('completep', 100): (5e-3, 1e-2, 2e-2, 4e-2), ('completep', 1000): (5e-3, 1e-2, 2e-2, 4e-2),
}


def smoke():
    """The heaviest configurations first: width 5120 and depth 1000."""
    return [width_job('mup', 5120, 1e-3), width_job('kaiming', 5120, 1.25e-4),
            depth_job('mup', 1000, 3e-4), depth_job('depth-mup', 1000, 1e-2), depth_job('completep', 1000, 1e-2)]


def full():
    jobs = [width_job(policy, width, lr) for (policy, width), lrs in WIDTH_LRS.items() for lr in lrs]
    return jobs + [depth_job(policy, depth, lr) for (policy, depth), lrs in DEPTH_LRS.items() for lr in lrs]


def rest():
    done = {job['name'] for job in smoke()}
    return [job for job in full() if job['name'] not in done]


def extra():
    """Standard muP was still improving at the top of its depth grids, so extend them upward."""
    return ([depth_job('mup', 100, lr) for lr in (2e-2, 4e-2)] +
            [depth_job('mup', 1000, lr) for lr in (1e-2, 2e-2, 4e-2)])


SWEEPS = {'pilot': pilot, 'smoke': smoke, 'rest': rest, 'extra': extra}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', choices=SWEEPS, required=True)
    parser.add_argument('--execute', action='store_true', help='Submit the runs to Modal.')
    args = parser.parse_args(argv)
    jobs = SWEEPS[args.sweep]()
    for job in jobs:
        print(job['name'])
    print(f'{len(jobs)} runs')
    if not args.execute:
        print('Preview only. Add --execute to train these configurations.')
        return
    data = MODAL_SHARED_DATASETS_DIR / DEFAULT_DATASET_DIR_NAME
    import modal
    with modal.enable_output():
        with app.run(name=timestamped_modal_app_name('a2-stress-sweep'), detach=True,
                     environment_name=MODAL_ENVIRONMENT):
            remote = _run.with_options(secrets=secrets(include_wandb=True), max_containers=MAX_PARALLEL_RUNS)
            for job in jobs:
                arguments = ['--train-path', str(data / 'train'), '--val-path', str(data / 'val'),
                             '--output', '/root/data/a2-stress/' + job['name'], *job['args']]
                call = remote.spawn(arguments)
                print(f"spawned {job['name']} ({call.object_id})")


if __name__ == '__main__':
    main()
