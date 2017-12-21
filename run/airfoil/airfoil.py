import click
import numpy as np
from nutils import function as fn, _, log, plot, matrix
from bbflow import cases, solvers, util, quadrature, reduction, ensemble as ens

from bbflow.cases.airfoil import rotmat


@click.group()
def main():
    pass


@util.pickle_cache('airfoil-{fast}-{piola}.case')
def get_case(fast: bool = False, piola: bool = False):
    case = cases.airfoil(override=fast, amax=35, nterms=7, piola=piola)
    case.restrict(viscosity=6.0)
    return case


@util.pickle_cache('airfoil-{piola}-{num}.ens')
def get_ensemble(fast: bool = False, piola: bool = False, num: int = 10):
    case = get_case(fast, piola)
    case.ensure_shareable()
    scheme = list(quadrature.full(case.ranges(), num))
    solutions = ens.make_ensemble(case, solvers.navierstokes, scheme, weights=True, parallel=fast)
    supremizers = ens.make_ensemble(
        case, solvers.supremizer, scheme, weights=False, parallel=True, args=[solutions],
    )
    return scheme, solutions, supremizers


@util.pickle_cache('airfoil-{piola}-{nred}.rcase')
def get_reduced(piola: bool = False, nred: int = 10, fast: int = None, num: int = None):
    case = get_case(fast, piola)
    scheme, solutions, supremizers = get_ensemble(fast, piola, num)

    eig_sol = reduction.eigen(case, solutions, fields=['v', 'p'])
    rb_sol, meta = reduction.reduced_bases(case, solutions, eig_sol, (nred, nred), meta=True)
    eig_sup = reduction.eigen(case, supremizers, fields=['v'])
    rb_sup = reduction.reduced_bases(case, supremizers, eig_sup, (nred,))

    reduction.plot_spectrum(
        [('solutions', eig_sol), ('supremizers', eig_sup)],
        plot_name=util.make_filename(get_reduced, 'airfoil-spectrum-{piola}', piola=piola),
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
        case, mu, lhs, colorbar=False, figsize=(10,10), fields=['v', 'p'],
        plot_name='full', index=index, axes=False
    )


@main.command()
@click.option('--angle', default=0.0)
@click.option('--velocity', default=1.0)
@click.option('--piola/--no-piola', default=False)
@click.option('--nred', '-r', default=10)
@click.option('--index', '-i', default=0)
def rsolve(angle, velocity, piola, nred, index):
    case = get_reduced(piola=piola, nred=nred)
    angle = -angle / 180 * np.pi
    mu = case.parameter(angle=angle, velocity=velocity)
    with util.time():
        try:
            lhs = solvers.navierstokes_block(case, mu)
        except AssertionError:
            lhs = solvers.navierstokes(case, mu)
    with util.time():
    solvers.plots(
        case, mu, lhs, colorbar=False, figsize=(10,10), fields=['v', 'p'],
        plot_name='red', index=index, axes=False
    )


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--num', '-n', default=8)
def ensemble(fast, piola, num):
    get_ensemble(fast, piola, num)


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--num', '-n', default=8)
@click.option('--nred', '-r', default=10)
def reduce(fast, piola, num, nred):
    get_reduced(piola, nred, fast, num)


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.argument('nred', nargs=-1, type=int)
def results(fast, piola, nred):
def _results(fast: bool = False, piola: bool = False, block: bool = False, nred=10):
    tcase = get_case(fast=fast, piola=piola)
    tcase.ensure_shareable()

    scheme = list(quadrature.full(tcase.ranges(), 5))
    ttime, tsol = ens.make_ensemble(tcase, solvers.navierstokes, scheme, parallel=True, return_time=True)

    if not piola:
        block = False

    res = []
    for nr in nred:
        rcase = get_reduced(piola=piola, nred=nr)
        solver = solvers.navierstokes_block if block else solvers.navierstokes
        rtime, rsol = ens.make_ensemble(rcase, solver, scheme, return_time=True)
        mu = tcase.parameter()
        verrs = ens.errors(tcase, rcase, tsol, rsol, tcase['v-h1s'](mu), scheme)
        perrs = ens.errors(tcase, rcase, tsol, rsol, tcase['p-l2'](mu), scheme)
        absf, relf = force_err(tcase, rcase, tsol, rsol, scheme)
        res.append([
            rcase.size // 3, rcase.meta['err-v'], rcase.meta['err-p'],
            *verrs, *perrs, *absf, *relf, ttime / rtime
        ])

    res = np.array(res)
    np.savetxt(
        util.make_filename(
            _results, 'airfoil-results-{piola}-{block}.csv',
            piola=piola, block=block,
        ),
        res
    )


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--block/--no-block', default=False)
@click.argument('nred', nargs=-1, type=int)
def results(fast, piola, block, nred):
    return _results(fast=fast, piola=piola, block=block, nred=nred)


if __name__ == '__main__':
    main()
