import click
import numpy as np
from itertools import islice, tee
from nutils import log, config
from aroma import cases, solvers, util, quadrature, reduction, ensemble as ens, visualization
from aroma.affine import NumpyArrayIntegrand
import multiprocessing
from tqdm import tqdm


@click.group()
def main():
    pass


@util.filecache('alecyl-{fast}-{piola}.case')
def get_case(fast: bool = False, piola: bool = False):
    case = cases.alecyl(piola=piola, rmin=0.5, rmax=20, nelems=150)
    case.restrict(velocity=1.0)
    case.precompute(force=fast)
    return case


@util.filecache('alecyl-{piola}.ens')
def get_ensemble(fast: bool = False, piola: bool = False, dt: float = 1e-2, nsteps: int = 100):
    case = get_case(fast, piola)
    case.restrict(viscosity=100.0)
    mu = case.parameter(velocity=1, viscosity=100)

    scheme = quadrature.full(case.ranges(ignore='time'), 1)
    ensemble = ens.Ensemble(scheme)
    ensemble.compute('solutions', case, solvers.navierstokes_time, time=True, kwargs={'dt': dt, 'nsteps': nsteps})
    ensemble.compute('supremizers', case, solvers.supremizer, parallel=False, args=[ensemble['solutions']])

    return ensemble


@util.filecache('alecyl-{piola}-{sups}-{nred}.rcase')
def get_reduced(piola: bool = False, sups: bool = True, nred: int = 10, fast: int = None):
    case = get_case(fast, piola)
    ensemble = get_ensemble(fast, piola)

    reducer = reduction.EigenReducer(case, ensemble)
    reducer.add_basis('v', parent='v', ensemble='solutions', ndofs=nred, norm='h1s')
    if sups:
        reducer.add_basis('s', parent='v', ensemble='supremizers', ndofs=nred, norm='h1s')
    reducer.add_basis('p', parent='p', ensemble='solutions', ndofs=nred, norm='l2')

    reducer.plot_spectra(util.make_filename(get_reduced, 'alecyl-spectrum-{piola}', piola=piola))
    return reducer()


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@util.common_args
def disp(fast, piola):
    case = get_case(fast, piola)
    print(case)


@main.command()
@click.option('--velocity', default=1.0)
@click.option('--viscosity', default=1.0)
@click.option('--nsteps', default=500)
@click.option('--timestep', default=0.5)
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--initsol', type=click.File('rb'), default=None)
@util.common_args
def solve(velocity, viscosity, nsteps, timestep, fast, piola, initsol):
    case = get_case(fast, piola)
    mu = case.parameter(velocity=velocity, viscosity=viscosity)

    with util.time():
        kwargs = {'initsol': np.load(initsol)} if initsol else {}
        timestepper = solvers.navierstokes_time(
            case, mu, maxit=10, nsteps=nsteps, dt=timestep, solver='mkl', tsolver='cn', **kwargs,
        )
        solutions = []
        for (mu, lhs) in timestepper:
            solutions.append(lhs)

    np.save('solutions.npy', np.array(solutions))


@main.command()
@click.option('--velocity', default=1.0)
@click.option('--viscosity', default=1.0)
@click.option('--piola/--no-piola', default=False)
@click.option('--nsteps', default=500)
@click.option('--timestep', default=0.5)
@click.option('--sups/--no-sups', default=True)
@click.option('--nred', '-r', default=10)
@util.common_args
def rsolve(velocity, viscosity, piola, nsteps, timestep, sups, nred):
    tcase = get_case(piola=piola, fast=True)
    case = get_reduced(piola=piola, sups=sups, nred=nred)
    mu = case.parameter(velocity=velocity, viscosity=viscosity)

    with util.time():
        solutions = solvers.navierstokes_time(case, mu, maxit=10, nsteps=nsteps, dt=timestep)

    # sol_ahead = solutions[-1500:]
    # sol_behind = solutions[-1501:]
    # forces = []
    # for i, ((_, lhs_prev), (mu, lhs_next)) in tqdm(enumerate(zip(sol_ahead, sol_behind))):
    #     lhs_next = case.solution_vector(lhs_next, tcase, mu, lift=True)
    #     lhs_prev = case.solution_vector(lhs_prev, tcase, mu, lift=True)
    #     forces.append(solvers.force(tcase, mu, lhs_next, lhs_prev, dt=timestep))

    for i, (mu, lhs) in enumerate(solutions[-1000:]):
        visualization.velocity(case, mu, lhs, name='pics/red', index=i, ndigits=5, streams=False)

    np.save(f'denseforces-{nred:02}', np.array(forces))


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--nsteps', default=500)
@click.option('--timestep', default=0.5)
@util.common_args
def ensemble(fast, piola, nsteps, timestep):
    get_ensemble(fast, piola, timestep, nsteps)


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--sups/--no-sups', default=True)
@click.option('--nred', '-r', default=10)
@util.common_args
def reduce(fast, piola, sups, nred):
    get_reduced(piola, sups, nred, fast)


def _bfuns():
    case = get_case(fast=True, piola=True)
    rcase = get_reduced(piola=True, sups=True, nred=80)

    rb_sol = {
        'v': rcase.projection[:80,:],
        's': rcase.projection[80:160,:],
        'p': rcase.projection[160:,:],
    }

    # eig_sol = reduction.eigen(case, solutions, fields=['v', 'p'])
    # rb_sol = reduction.reduced_bases(case, solutions, eig_sol, (12, 12))
    # eig_sup = reduction.eigen(case, supremizers, fields=['v'])
    # rb_sup = reduction.reduced_bases(case, supremizers, eig_sup, (12,))

    for i in range(6):
        visualization.velocity(
            case, case.parameter(), rb_sol['v'][i], colorbar=False, figsize=(10,10), fields=['v'],
            axes=False, density=2, lift=False, name='bfun-v-piola', index=i,
        )
        # visualization.pressure(
        #     case, case.parameter(), rb_sol['p'][i], colorbar=False, figsize=(10,10), fields=['p'],
        #     axes=False, lift=False, name='bfun-p-piola', index=i,
        # )
        # visualization.velocity(
        #     case, case.parameter(), rb_sol['s'][i], colorbar=False, figsize=(10,10), fields=['v'],
        #     axes=False, density=2, lift=False, name='bfun-s-piola', index=i,
        # )


@main.command()
@util.common_args
def bfuns():
    _bfuns()


@util.filecache('alecyl-{piola}-{sups}.rens')
def _results(fast: bool = False, piola: bool = False, sups: bool = False, block: bool = False, nred=[10]):
    tcase = get_case(fast=fast, piola=piola)
    tcase.ensure_shareable()

    ensemble = get_ensemble(fast=fast, piola=piola, num=8)

    tcase.restrict(time=0.0)
    scheme = quadrature.sparse(tcase.ranges(), 8)
    tcase.restrict(time=None)

    if not piola:
        block = False

    for nr in nred:
        _ensemble = ens.Ensemble(scheme)
        args = [[0.05] * len(scheme), [500] * len(scheme)]

        rcase = get_reduced(piola=piola, sups=sups, nred=nr)
        rtime = _ensemble.compute(f'rsol-{nr}', rcase, solvers.navierstokes_time, time=True, parallel=False, args=args)

        ensemble[f'rsol-{nr}'] = _ensemble[f'rsol-{nr}']

    return ensemble


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--sups/--no-sups', default=True)
@click.option('--block/--no-block', default=False)
@click.argument('nred', nargs=-1, type=int)
@util.common_args
def results(fast, piola, sups, block, nred):
    return _results(fast=fast, piola=piola, sups=sups, block=block, nred=nred)


def _new_postprocess():
    case = get_case(fast=True, piola=False)
    mu = case.parameter(velocity=1, viscosity=100)

    solutions = np.load('solutions.npy')
    cons = case.constraints

    forces = []
    lifted = map(lambda g: case.solution_vector(g, mu, lift=True), solutions)

    lift1, lift2 = tee(lifted, 2)
    L = len(solutions)
    current = islice(lift1, 1, L)
    behind = islice(lift2, 0, L-1)
    for i, (lhs, lhs_prev) in enumerate(zip(current, behind)):
        rforce = solvers.force(case, mu, lhs, prev=lhs_prev, dt=0.025, method='recover', tsolver='cn')
        forces.append([mu['time'], *rforce])
        print(i)
        # visualization.velocity(case, mu, lhs, name='pics/full', axes=False,
        #                        colorbar=True, index=i, ndigits=4, streams=False, lift=False)
        # visualization.pressure(case, mu, lhs, name='pics/full', axes=False,
        #                        colorbar=True, index=i, ndigits=4, lift=False)

    np.save('forces.npy', np.array(forces))


@main.command()
@util.common_args
def postprocess():
    _new_postprocess()


@main.command()
@util.common_args
def investigate():
    case = get_case(piola=False, fast=True)

    # Indices of the different Dirichlet boundaries
    dirichlet_inner_x = np.arange(200)
    dirichlet_outer_x = np.arange(40048, 40153)
    dirichlet_inner_y = np.arange(40200, 40400)
    dirichlet_outer_y = np.arange(80248, 80353)
    free = np.where(np.isnan(case.constraints))

    # Verify that they are sensible
    mu = case.parameter(velocity=1, viscosity=100)
    lift = case.lift(mu)
    assert np.all(case.constraints[dirichlet_inner_x] == 0.0)
    assert np.all(case.constraints[dirichlet_outer_x] == 0.0)
    assert np.all(case.constraints[dirichlet_inner_y] == 0.0)
    assert np.all(case.constraints[dirichlet_outer_y] == 0.0)
    assert np.all(lift[dirichlet_inner_x] == 0.0)
    assert np.allclose(lift[dirichlet_outer_x], 1.0)
    assert np.all(lift[dirichlet_inner_y] == 0.0)
    assert np.all(lift[dirichlet_outer_y] == 0.0)
    assert np.all(np.where(~np.isnan(case.constraints)) == np.hstack((
        dirichlet_inner_x, dirichlet_outer_x, dirichlet_inner_y, dirichlet_outer_y,
    )))

    # Load computed solution vectors
    solutions = np.load('solutions.npy')
    for sln_prev, sln_next in zip(solutions[:-1], solutions[1:]):
        sln_prev = sln_prev + lift
        sln_next = sln_next + lift
        dt = 0.05

        m_next = case['v-l2'](mu, cont=(None, sln_next))
        m_prev = case['v-l2'](mu, cont=(None, sln_prev))
        m = (m_next - m_prev) / dt
        a_next = case['laplacian'](mu, cont=(None, sln_next))
        c_next = case['convection'](mu, cont=(None, sln_next, sln_next))
        a_prev = case['laplacian'](mu, cont=(None, sln_prev))
        c_prev = case['convection'](mu, cont=(None, sln_prev, sln_prev))
        b = case['divergence'](mu, cont=(None, sln_next))
        check = m + (a_next + c_next)/2 + (a_prev + c_prev)/2 + b

        print(np.max(check[free]))

        if np.allclose(check[free], 0.0, atol=5e-8):
            print('ok!')
        else:
            print('not ok!')

    # local_force = case['force'](mu, cont=(sln_next, None))
    # print(local_force)

    # wx = case['xforce'](mu)
    # global_force_x = (
    #     case['v-l2'](mu, cont=(wx, sln_next)) / dt -
    #     case['v-l2'](mu, cont=(wx, sln_prev)) / dt +
    #     case['divergence'](mu, cont=(wx, sln_next)) +
    #     case['laplacian'](mu, cont=(wx, sln_next)) +
    #     case['convection'](mu, cont=(wx, sln_next, sln_next))
    # )

    # wy = case['yforce'](mu)
    # global_force_y = (
    #     case['v-l2'](mu, cont=(wy, sln_next)) / dt -
    #     case['v-l2'](mu, cont=(wy, sln_prev)) / dt +
    #     case['divergence'](mu, cont=(wy, sln_next)) +
    #     case['laplacian'](mu, cont=(wy, sln_next)) +
    #     case['convection'](mu, cont=(wy, sln_next, sln_next))
    # )

    # print(global_force_x)
    # # print(global_force_y)


if __name__ == '__main__':
    main()
