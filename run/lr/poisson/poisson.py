import click
from tempfile import TemporaryDirectory
from pathlib import Path
from jinja2 import Template
from splipy import Surface
from splipy.io import G2
from subprocess import run, PIPE
from tqdm import tqdm
import h5py
import lrspline as lr
import numpy as np
from aroma import util, quadrature, case, ensemble as ens, cases, solvers, reduction
from nutils import log


@click.group()
def main():
    pass


def ifem_solve(root, mu, i):
    with open('case.xinp') as f:
        template = Template(f.read())
    with open(root / 'case.xinp', 'w') as f:
        f.write(template.render(geometry='square.g2', **mu))

    patch = Surface()
    patch.set_dimension(3)
    with G2(str(root / 'square.g2')) as g2:
        g2.write(patch)
        try:
            g2.fstream.close()
        except:
            pass

    result = run(['/home/eivind/repos/IFEM/Apps/Poisson/build/bin/Poisson',
                  'case.xinp', '-adap', '-hdf5'], cwd=root, stdout=PIPE, stderr=PIPE)
    result.check_returncode()

    with h5py.File(root / 'case.hdf5', 'r') as h5:
        final = str(len(h5) - 1)
        group = h5[final]['Poisson-1']
        patchbytes = group['basis']['1'][:].tobytes()
        geompatch = lr.LRSplineSurface(patchbytes)
        coeffs = group['fields']['u']['1'][:]
        solpatch = geompatch.clone()
        solpatch.controlpoints = coeffs.reshape(len(solpatch), -1)

        with open(f'poisson-mesh-single-{i}.ps', 'wb') as f:
            geompatch.write_postscript(f)

        return geompatch, solpatch


def move_meshlines(source, target):
    """Duplicate meshlines in SOURCE to TARGET."""
    for meshline in source.meshlines:
        target.insert(meshline)
    target.generate_ids()


@util.filecache('poisson-{num}.case')
def get_case(num: int):
    with open(f'poisson-mesh-{num}.lr', 'rb') as f:
        mesh = lr.LRSplineSurface(f)
    case = cases.lrpoisson(mesh)
    return case


@util.filecache('poisson-{num}.ens')
def get_ensemble(num: int):
    scheme = quadrature.full([(0.25, 0.75), (0.25, 0.75)], num)

    # HACK
    solutions = []
    with TemporaryDirectory() as path:
        for i, (_, *mu) in enumerate(tqdm(scheme)):
            mu = dict(zip(['xcenter', 'ycenter'], mu))
            solutions.append(ifem_solve(Path(path), mu, i))

    rootpatch = solutions[0][0].clone()
    for patch, _ in solutions:
        move_meshlines(patch, rootpatch)

    for geompatch, solpatch in solutions:
        move_meshlines(rootpatch, geompatch)
        move_meshlines(rootpatch, solpatch)

    for geompatch, _ in solutions:
        assert np.allclose(geompatch.controlpoints, rootpatch.controlpoints)

    with open(f'poisson-mesh-{num}.lr', 'wb') as f:
        rootpatch.write(f)
    with open(f'poisson-mesh-{num}.ps', 'wb') as f:
        rootpatch.write_postscript(f)

    ensemble = ens.Ensemble(scheme)
    ensemble['solutions'] = np.array([
        solpatch.controlpoints for _, solpatch in solutions
    ]).reshape((len(solutions), -1))

    return ensemble


@util.filecache('poisson-{num}-{nred}.rcase')
def get_reduced(num: int, nred: int):
    case = get_case(num)
    ens = get_ensemble(num)

    reducer = reduction.EigenReducer(case, ens)
    reducer.add_basis('u', parent='u', ensemble='solutions', ndofs=nred, norm='h1s')

    reducer.plot_spectra(util.make_filename(get_reduced, 'poisson-spectrum-{num}', num=num, nred=nred))
    return reducer(tol=1e-5, nrules=4)


@main.command()
@click.option('--xcenter', '-x', default=0.5)
@click.option('--ycenter', '-y', default=0.5)
@click.option('--num', '-n', default=8)
@util.common_args
def solve(xcenter, ycenter, num):
    case = get_case(num)
    mu = case.parameter(xcenter=xcenter, ycenter=ycenter)
    with util.time():
        lhs = solvers.poisson(case, mu)


@main.command()
@click.option('--xcenter', '-x', default=0.5)
@click.option('--ycenter', '-y', default=0.5)
@click.option('--num', '-n', default=8)
@click.option('--nred', default=10)
@util.common_args
def rsolve(xcenter, ycenter, num, nred):
    hcase = get_case(num)
    rcase = get_reduced(num, nred)
    mu = rcase.parameter(xcenter=xcenter, ycenter=ycenter)

    with util.time():
        lhs = solvers.poisson(rcase, mu)


@main.command()
@click.option('--num', '-n', default=8)
@util.common_args
def ensemble(num):
    get_ensemble(num)


@main.command()
@click.option('--num', '-n', default=8)
@util.common_args
def disp(num):
    print(get_case(num))


@main.command()
@click.option('--num', '-n', default=8)
@click.option('--nred', default=10)
@util.common_args
def reduce(num, nred):
    get_reduced(num, nred)


def _results(num: int, nred: int):
    hcase = get_case(num)

    scheme = quadrature.uniform(hcase.ranges(), 10)
    ensemble = ens.Ensemble(scheme)
    ensemble.compute('solutions', hcase, solvers.poisson, parallel=False)

    res = []
    for nr in nred:
        rcase = get_reduced(num, nr)
        rtime = ensemble.compute('rsol', rcase, solvers.poisson, parallel=False)
        mu = hcase.parameter()
        errs = ensemble.errors(hcase, 'solutions', rcase, 'rsol', hcase['u-h1s'](mu))
        res.append([rcase.size, rcase.meta['err-u'], *errs, rtime])

    np.savetxt(
        util.make_filename(
            _results, 'poisson-results-{num}.csv',
            num=num, nred=nred,
        ),
        np.array(res),
    )


@main.command()
@click.option('--num', '-n', default=8)
@click.argument('nred', nargs=-1, type=int)
@util.common_args
def results(num, nred):
    _results(num, nred)


if __name__ == '__main__':
    main()
