import click
from nutils import log, config
import multiprocessing
import numpy as np

from aroma import cases, solvers, util, quadrature, reduction, ensemble as ens


@util.pickle_cache('tshape-{fast}.case')
def get_case(fast: bool = False):
    return cases.tshape(override=fast)


@util.pickle_cache('tshape-{num}.ens')
def get_ensemble(num: int = 10, fast: bool = False):
    case = get_case(fast)
    scheme = list(quadrature.full(case.ranges(), num))
    solutions = ens.make_ensemble(case, solvers.navierstokes, scheme, weights=True, parallel=fast)
    supremizers = ens.make_ensemble(
        case, solvers.supremizer, scheme, weights=False, parallel=True, args=[solutions],
    )
    return scheme, solutions, supremizers


@util.pickle_cache('tshape-{nred}.rcase')
def get_reduced(nred: int = 10, fast: bool = False, num: int = 10):
    case = get_case(fast)
    scheme, solutions, supremizers = get_ensemble(num, fast)

    eig_sol = reduction.eigen(case, solutions, fields=['v', 'p'])
    rb_sol = reduction.reduced_bases(case, solutions, eig_sol, (nred, nred))
    eig_sup = reduction.eigen(case, supremizers, fields=['v'])
    rb_sup = reduction.reduced_bases(case, supremizers, eig_sup, (nred,))

    reduction.plot_spectrum(
        [('solutions', eig_sol), ('supremizers', eig_sup)],
        plot_name='tshape-spectrum', formats=['png', 'csv'],
    )

    projcase = reduction.make_reduced(case, rb_sol, rb_sup)
    return projcase


@click.group()
def main():
    pass


@main.command()
@click.option('--fast/--no-fast', default=False)
def disp(fast):
    case = get_case(fast)
    print(case)


@main.command()
@click.option('--viscosity', default=1.0)
@click.option('--velocity', default=1.0)
@click.option('--height', default=1.0)
@click.option('--fast/--no-fast', default=False)
def solve(viscosity, velocity, height, fast):
    case = get_case(fast)
    mu = case.parameter(viscosity=viscosity, velocity=velocity, height=height)
    with util.time():
        lhs = solvers.navierstokes(case, mu)
    solvers.plots(case, mu, lhs, colorbar=True, figsize=(10,10), fields=['v','p'], plot_name='full')


@main.command()
@click.option('--viscosity', default=1.0)
@click.option('--velocity', default=1.0)
@click.option('--height', default=1.0)
@click.option('--nred', '-r', default=10)
def rsolve(viscosity, velocity, height, nred):
    case = get_reduced(nred=nred)
    mu = case.parameter(viscosity=viscosity, velocity=velocity, height=height)
    with util.time():
        lhs = solvers.navierstokes(case, mu)
    solvers.plots(case, mu, lhs, colorbar=True, figsize=(10,10), fields=['v', 'p'], plot_name='red')


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--num', '-n', default=8)
def ensemble(fast, num):
    get_ensemble(num, fast)


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--num', '-n', default=8)
@click.option('--nred', '-r', default=10)
def reduce(fast, num, nred):
    get_reduced(nred, fast, num)


if __name__ == '__main__':
    with config(nprocs=multiprocessing.cpu_count()):
        main()
