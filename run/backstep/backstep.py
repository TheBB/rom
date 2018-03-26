import click
from nutils import log, config
import multiprocessing

from aroma import cases, solvers, util, quadrature, reduction, ensemble as ens, visualization


@util.pickle_cache('backstep-{fast}.case')
def get_case(fast: bool = False):
    return cases.backstep(override=fast)


@util.pickle_cache('backstep-{num}.ens')
def get_ensemble(num: int = 10, fast: bool = False):
    case = get_case(fast)
    case.ensure_shareable()
    scheme = list(quadrature.sparse(case.ranges(), num))
    solutions = ens.make_ensemble(case, solvers.navierstokes, scheme, weights=True, parallel=fast)
    supremizers = ens.make_ensemble(
        case, solvers.supremizer, scheme, weights=False, parallel=True, args=[solutions],
    )
    return scheme, solutions, supremizers


@util.pickle_cache('backstep-{nred}.rcase')
def get_reduced(nred: int = 10, fast: bool = False, num: int = 10):
    case = get_case(fast)
    scheme, solutions, supremizers = get_ensemble(num, fast)

    reducer = reduction.EigenReducer(case, solutions=solutions, supremizers=supremizers)
    reducer.add_basis('v', parent='v', ensemble='solutions', ndofs=nred, norm='h1s')
    reducer.add_basis('s', parent='v', ensemble='supremizers', ndofs=nred, norm='h1s')
    reducer.add_basis('p', parent='p', ensemble='solutions', ndofs=nred, norm='l2')

    return reducer()


@click.group()
def main():
    pass


@main.command()
@click.option('--fast/--no-fast', default=False)
def disp(fast):
    print(get_case(fast))


@main.command()
@click.option('--viscosity', default=20.0)
@click.option('--velocity', default=1.0)
@click.option('--height', default=1.0)
@click.option('--length', default=10.0)
@click.option('--fast/--no-fast', default=False)
def solve(viscosity, velocity, height, length, fast):
    case = get_case(fast)
    mu = case.parameter(viscosity=viscosity, velocity=velocity, height=height, length=length)
    with util.time():
        lhs = solvers.navierstokes(case, mu)
    visualization.velocity(case, mu, lhs, figsize=(15,3), name='full')
    visualization.pressure(case, mu, lhs, figsize=(15,3), name='full')


@main.command()
@click.option('--viscosity', default=20.0)
@click.option('--velocity', default=1.0)
@click.option('--height', default=1.0)
@click.option('--length', default=10.0)
@click.option('--nred', '-r', default=10)
def rsolve(viscosity, velocity, height, length, nred):
    case = get_reduced(nred=nred)
    mu = case.parameter(viscosity=viscosity, velocity=velocity, height=height, length=length)
    with util.time():
        lhs = solvers.navierstokes(case, mu)
    visualization.velocity(case, mu, lhs, figsize=(15,3), name='red')
    visualization.pressure(case, mu, lhs, figsize=(15,3), name='red')


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
