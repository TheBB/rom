import click
import numpy as np
from nutils import function as fn, _, log, plot, matrix, config
from os import path
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import multiprocessing

from aroma import cases, solvers, util, quadrature, reduction, ensemble as ens
from aroma.cases.airfoil import rotmat


@click.group()
def main():
    pass


@util.pickle_cache('airfoil-{fast}-{piola}.case')
def get_case(fast: bool = False, piola: bool = False):
    case = cases.airfoil(override=fast, amax=25, piola=piola, fname='NACA64', cylrot=0.1, rmax=15, nelems=35)
    case.restrict(viscosity=1.0)
    return case


@util.pickle_cache('airfoil-{piola}-{imported}.ens')
def get_ensemble(fast: bool = False, piola: bool = False, imported: bool = False):
    if imported:
        return import_ensemble(piola=piola)
    case = get_case(fast, piola)
    case.ensure_shareable()
    scheme = list(quadrature.uniform([(-25*np.pi/180, 25*np.pi/180), (2, 20)], (11, 10)))
    solutions = ens.make_ensemble(case, solvers.navierstokes, scheme, weights=True, parallel=False)
    supremizers = ens.make_ensemble(
        case, solvers.supremizer, scheme, weights=False, parallel=False, args=[solutions],
    )
    return scheme, solutions, supremizers


def import_ensemble(piola: bool = False):
    case = get_case(fast=True, piola=piola)
    case.ensure_shareable()
    scheme, __, __ = get_ensemble(piola=piola, imported=False)
    solutions = []

    for (angle, velocity), weight in scheme:
        param = case.parameter(angle, velocity)
        angle = int(np.round(-angle / np.pi * 180))
        angle = str(angle) if angle >= 0 else str(abs(angle)) + '-'
        velocity = str(int(velocity))

        filename = f'openfoam/{angle}/{velocity}/{angle}{velocity}.vtk'
        log.user(filename)
        reader = vtk.vtkDataSetReader()
        reader.SetFileName(filename)
        reader.Update()
        dataset = reader.GetOutput()
        pointdata = dataset.GetPointData()

        velocity = vtk_to_numpy(pointdata.GetAbstractArray('U'))
        pressure = vtk_to_numpy(pointdata.GetAbstractArray('p'))
        npts = len(velocity) // 2
        velocity = velocity[:npts]
        pressure = pressure[:npts]

        linbasis = case.domain.basis('spline', degree=1)
        vsol = linbasis.dot(velocity[:,0])[_] * (1,0) + linbasis.dot(velocity[:,1])[_] * (0,1)
        psol = linbasis.dot(pressure)
        geom = case.physical_geometry(param)

        vbasis = case.basis('v', param)
        vgrad = vbasis.grad(geom)
        pbasis = case.basis('p', param)
        lhs = case.domain.project(psol, onto=pbasis, geometry=geom, ischeme='gauss9')

        vind = case.basis_indices('v')
        itg = fn.outer(vbasis).sum([-1]) + fn.outer(vgrad).sum([-1,-2])
        mx = case.domain.integrate(itg, geometry=geom, ischeme='gauss9').core[np.ix_(vind,vind)]
        itg = (vbasis * vsol[_,:]).sum([-1]) + (vgrad * vsol.grad(geom)[_,:,:]).sum([-1,-2])
        rhs = case.domain.integrate(itg, geometry=geom, ischeme='gauss9')[vind]
        lhs[vind] = matrix.ScipyMatrix(mx).solve(rhs)

        lift = case._lift(param)
        solutions.append((lhs - lift) * weight)

    solutions = np.array(solutions)
    supremizers = ens.make_ensemble(
        case, solvers.supremizer, scheme, weights=False, parallel=False, args=[solutions],
    )
    return scheme, solutions, supremizers


@util.pickle_cache('airfoil-{piola}-{imported}-{nred}.rcase')
def get_reduced(piola: bool = False, nred: int = 10, imported: bool = False, fast: bool = False):
    case = get_case(fast, piola)
    scheme, solutions, supremizers = get_ensemble(fast, piola, imported)

    eig_sol = reduction.eigen(case, solutions, fields=['v', 'p'])
    rb_sol, meta = reduction.reduced_bases(case, solutions, eig_sol, (nred, nred), meta=True)
    eig_sup = reduction.eigen(case, supremizers, fields=['v'])
    rb_sup = reduction.reduced_bases(case, supremizers, eig_sup, (nred,))

    reduction.plot_spectrum(
        [('solutions', eig_sol), ('supremizers', eig_sup)],
        plot_name=util.make_filename(
            get_reduced, 'airfoil-spectrum-{piola}-{imported}', piola=piola, imported=imported
        ),
        formats=['png', 'csv'],
    )

    projcase = reduction.make_reduced(case, rb_sol, rb_sup, meta=meta)

    if piola:
        with log.context('block project'):
            with log.context('convection'):
                projcase['convection-vvv'] = case['convection'].project(rb_sol['v'])
                projcase['convection-svv'] = case['convection'].project((rb_sup['v'], rb_sol['v'], rb_sol['v']))
            with log.context('laplacian'):
                projcase['laplacian-vv'] = case['laplacian'].project(rb_sol['v'])
                projcase['laplacian-sv'] = case['laplacian'].project((rb_sup['v'], rb_sol['v']))
            with log.context('divergence'):
                projcase['divergence-sp'] = case['divergence'].project((rb_sup['v'], rb_sol['p']))

    return projcase


def force_err(hicase, locase, hifi, lofi, scheme):
    abs_err, rel_err = np.zeros(2), np.zeros(2)
    for hilhs, lolhs, (mu, weight) in zip(hifi, lofi, scheme):
        mu = locase.parameter(*mu)
        hiforce = hicase['force'](mu, cont=(hilhs,None))
        loforce = locase['force'](mu, cont=(lolhs,None))
        err = np.abs(hiforce - loforce)
        abs_err += weight * err
        rel_err += weight * err / np.abs(hiforce)

    abs_err /= sum(w for __, w in scheme)
    rel_err /= sum(w for __, w in scheme)
    return abs_err, rel_err


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
def disp(fast, piola):
    print(get_case(fast, piola))


@main.command()
@click.option('--angle', default=0.0)
@click.option('--velocity', default=1.0)
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--index', '-i', default=0)
def solve(angle, velocity, fast, piola, index):
    case = get_case(fast, piola)
    angle = -angle / 180 * np.pi
    mu = case.parameter(angle=angle, velocity=velocity)
    with util.time():
        lhs = solvers.navierstokes(case, mu)
    solvers.plots(
        case, mu, lhs, colorbar=True, figsize=(10,10), fields=['v', 'p'],
        plot_name='full', index=index, axes=False
    )


@main.command()
@click.option('--angle', default=0.0)
@click.option('--velocity', default=1.0)
@click.option('--piola/--no-piola', default=False)
@click.option('--imported/--no-imported', default=False)
@click.option('--nred', '-r', default=10)
@click.option('--index', '-i', default=0)
def rsolve(angle, velocity, piola, imported, nred, index):
    case = get_reduced(piola=piola, nred=nred, imported=imported)
    angle = -angle / 180 * np.pi
    mu = case.parameter(angle=angle, velocity=velocity)
    with util.time():
        try:
            lhs = solvers.navierstokes_block(case, mu)
        except AssertionError:
            lhs = solvers.navierstokes(case, mu)
    solvers.plots(
        case, mu, lhs, colorbar=True, figsize=(10,10), fields=['v', 'p'],
        plot_name='red', index=index, axes=False
    )


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--imported/--no-imported', default=False)
def ensemble(fast, piola, imported):
    get_ensemble(fast, piola, imported)


@main.command()
@click.option('--piola/--no-piola', default=False)
@click.option('--num', '-n', default=0)
def compare(piola, num):
    scheme, fem_sol, fem_sup = get_ensemble(piola=piola, imported=False)
    __, fvm_sol, fvm_sup = get_ensemble(piola=piola, imported=True)
    case = get_case(fast=True, piola=piola)

    mu, weight = scheme[num]
    mu = case.parameter(*mu)
    fem_sol = fem_sol[num] / weight
    fvm_sol = fvm_sol[num] / weight

    solvers.plots(case, mu, fem_sol, plot_name='fem', index=num, colorbar=True, fields=['v'])
    solvers.plots(case, mu, fvm_sol, plot_name='fvm', index=num, colorbar=True, fields=['v'])
    solvers.plots(case, mu, fem_sol, plot_name='fem', index=num, colorbar=True, fields=['p'],
                  xlim=(-1,1), ylim=(-1,1), clim=(-20,20))
    solvers.plots(case, mu, fvm_sol, plot_name='fvm', index=num, colorbar=True, fields=['p'],
                  xlim=(-1,1), ylim=(-1,1), clim=(-20,20))


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--imported/--no-imported', default=False)
@click.option('--nred', '-r', default=10)
def reduce(fast, piola, imported, nred):
    get_reduced(piola, nred, imported, fast)


def _bfuns(fast: bool = False, piola: bool = False, imported: bool = False):
    __, solutions, supremizers = get_ensemble(fast, piola, imported)
    case = get_case(fast, piola)

    eig_sol = reduction.eigen(case, solutions, fields=['v', 'p'])
    rb_sol = reduction.reduced_bases(case, solutions, eig_sol, (12, 12))
    eig_sup = reduction.eigen(case, supremizers, fields=['v'])
    rb_sup = reduction.reduced_bases(case, supremizers, eig_sup, (12,))

    kw = {'piola': piola, 'imported': imported}

    for i in range(6):
        solvers.plots(
            case, case.parameter(), rb_sol['v'][i], colorbar=False, figsize=(13,8), fields=['v'],
            axes=False, xlim=(-4,9), ylim=(-4,4), density=3, lift=False,
            plot_name=util.make_filename(_bfuns, 'bfun-v-{piola}-{imported}', **kw), index=i,
        )
        solvers.plots(
            case, case.parameter(), rb_sol['p'][i], colorbar=False, figsize=(10,10), fields=['p'],
            axes=False, xlim=(-2,2), ylim=(-2,2), lift=False,
            plot_name=util.make_filename(_bfuns, 'bfun-p-{piola}-{imported}', **kw), index=i,
        )
        solvers.plots(
            case, case.parameter(), rb_sup['v'][i], colorbar=False, figsize=(10,10), fields=['v'],
            axes=False, density=2, lift=False,
            plot_name=util.make_filename(_bfuns, 'bfun-s-{piola}-{imported}', **kw), index=i,
        )


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--imported/--no-imported', default=False)
def bfuns(fast, piola, imported):
    _bfuns(fast=fast, piola=piola, imported=imported)


@main.command('compare-ensembles')
@click.option('--piola/--no-piola', default=False)
def compare_ensembles(piola):
    case = get_case(piola=piola, fast=True)
    mu = case.parameter()
    vh1 = case['v-h1s'](mu)
    pl2 = case['p-l2'](mu)

    __, orig_sol, orig_sup = get_ensemble(piola=piola, imported=False)
    __, impo_sol, impo_sup = get_ensemble(piola=piola, imported=True)

    verrs, perrs = 0.0, 0.0

    for i, (ov, iv) in enumerate(zip(orig_sol, impo_sol)):
        diff = ov - iv
        verrs += np.sqrt(vh1.matvec(diff).dot(diff) / vh1.matvec(ov).dot(ov))
        perrs += np.sqrt(pl2.matvec(diff).dot(diff) / pl2.matvec(ov).dot(ov))

    verrs /= len(orig_sol)
    perrs /= len(impo_sol)
    log.user(f'{verrs: >10.2e} {perrs: >10.2e}')


def _results(fast: bool = False, piola: bool = False, block: bool = False, imported: bool = False, nred=10):
    tcase = get_case(fast=fast, piola=piola)
    tcase.ensure_shareable()

    scheme, tsol, __ = get_ensemble(piola=piola, imported=imported)
    tsol = np.array([ts/wt for ts, (__, wt) in zip(tsol, scheme)])

    # scheme = list(quadrature.full([(-25*np.pi/180, 25*np.pi/180), (2, 20)], 2))
    # ttime, tsol = ens.make_ensemble(tcase, solvers.navierstokes, scheme, parallel=True, return_time=True)

    if not piola:
        block = False

    res = []
    for nr in nred:
        rcase = get_reduced(piola=piola, imported=imported, nred=nr)
        solver = solvers.navierstokes_block if block else solvers.navierstokes
        rtime, rsol = ens.make_ensemble(rcase, solver, scheme, return_time=True)
        mu = tcase.parameter()
        verrs = ens.errors(tcase, rcase, tsol, rsol, tcase['v-h1s'](mu), scheme)
        perrs = ens.errors(tcase, rcase, tsol, rsol, tcase['p-l2'](mu), scheme)
        absf, relf = force_err(tcase, rcase, tsol, rsol, scheme)
        res.append([
            rcase.size // 3, rcase.meta['err-v'], rcase.meta['err-p'],
            *verrs, *perrs, *absf, *relf, rtime
        ])

    res = np.array(res)
    np.savetxt(
        util.make_filename(
            _results, 'airfoil-results-{piola}-{block}-{imported}.csv',
            piola=piola, block=block, imported=imported,
        ),
        res
    )


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--block/--no-block', default=False)
@click.option('--imported/--no-imported', default=False)
@click.argument('nred', nargs=-1, type=int)
def results(fast, piola, block, imported, nred):
    return _results(fast=fast, piola=piola, block=block, imported=imported, nred=nred)


if __name__ == '__main__':
    with config(nprocs=multiprocessing.cpu_count()):
        main()
