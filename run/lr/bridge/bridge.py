from tempfile import TemporaryDirectory
import shutil
from jinja2 import Template
from subprocess import run, PIPE
from tqdm import tqdm
from pathlib import Path

from aroma import util, quadrature, case, ensemble as ens, cases, solvers, reduction


IFEM = '/home/eivind/repos/IFEM/Apps/Elasticity/Linear/build/bin/LinEl'


def ifem_solve(root, mu, i, order):
    order = {2: 'linear', 3: 'quadratic', 4: 'cubic'}[order]
    with open(f'{order}/bridge-{order}.xinp', 'r') as f:
        template = Template(f.read())
    with open(root / f'bridge-{order}.xinp', 'w') as f:
        f.write(template.render(geometry='bridge-{order}.g2', **mu))

    shutil.copy(f'{order}/bridge-topology.xinp', root / 'bridge-topology.xinp')
    shutil.copy(f'{order}/bridge-topologysets.xinp', root / 'bridge-topologysets.xinp')
    shutil.copy(f'{order}/bridge-{order}.g2', root / 'bridge-{order}.g2')

    result = run([IFEM, f'bridge-{order}.xinp', '-adap1', '-cgl2', '-hdf5'],
                 cwd=root, stdout=PIPE, stderr=PIPE)
    result.check_returncode()

    shutil.copy(root / f'bridge-{order}.hdf5', f'{order}/results/{i:02}.hdf5')


def get_ensemble(num: int):
    scheme = quadrature.full([(-97.175, 97.175)], num)

    with TemporaryDirectory() as path:
        for i, (_, *mu) in enumerate(tqdm(scheme)):
            mu = dict(zip(['center'], mu))
            ifem_solve(Path(path), mu, i, order=2)


if __name__ == '__main__':
    get_ensemble(2)
