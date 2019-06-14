import click
import numpy as np
from nutils import log, config
from aroma import cases, solvers, util, quadrature, reduction, ensemble as ens, visualization
from aroma.affine import NumpyArrayIntegrand
import multiprocessing


@click.group()
def main():
    pass


@util.filecache('alecyl-{fast}-{piola}.case')
def get_case(fast: bool = False, piola: bool = False):
    case = cases.alecyl(piola=piola, rmin=0.5, rmax=40, nelems=70)
    case.restrict(velocity=1.0)
    case.precompute(force=fast)
    return case


@util.filecache('alecyl-{piola}-{num}.ens')
def get_ensemble(fast: bool = False, piola: bool = False, num: int = 10):
    case = get_case(fast, piola)
    case.ensure_shareable()

    # Temporarily restrict time
    case.restrict(time=0.0)
    scheme = quadrature.sparse(case.ranges(), num)
    case.restrict(time=None)

    ensemble = ens.Ensemble(scheme)
    args = [[0.05] * len(scheme), [500] * len(scheme)]
    ensemble.compute('solutions', case, solvers.navierstokes_time, parallel=fast, time=True, args=args)
    ensemble.compute('supremizers', case, solvers.supremizer, parallel=True, args=[ensemble['solutions']])
    return ensemble


@util.filecache('alecyl-{piola}-{sups}-{nred}.rcase')
def get_reduced(piola: bool = False, sups: bool = True, nred: int = 10, fast: int = None, num: int = None):
    case = get_case(fast, piola)
    ensemble = get_ensemble(fast, piola, num)

    reducer = reduction.EigenReducer(case, ensemble)
    reducer.add_basis('v', parent='v', ensemble='solutions', ndofs=nred, norm='h1s')
    if sups:
        reducer.add_basis('s', parent='v', ensemble='supremizers', ndofs=nred, norm='h1s')
    reducer.add_basis('p', parent='p', ensemble='solutions', ndofs=nred, norm='l2')

    # if sups and piola:
    #     reducer.override('convection', 'vvv', 'svv', soft=True)
    #     reducer.override('laplacian', 'vv', 'sv', soft=True)
    #     reducer.override('divergence', 'sp', soft=True)

    reducer.plot_spectra(util.make_filename(get_reduced, 'alecyl-spectrum-{piola}', piola=piola))
    return reducer()


# def force_err(hicase, locase, hifi, lofi, scheme):
#     abs_err, rel_err = np.zeros(2), np.zeros(2)
#     max_abs_err, max_rel_err = np.zeros(2), np.zeros(2)
#     for hilhs, lolhs, (mu, weight) in zip(hifi, lofi, scheme):
#         mu = locase.parameter(*mu)
#         hiforce = hicase['force'](mu, cont=(hilhs,None))
#         loforce = locase['force'](mu, cont=(lolhs,None))
#         aerr = np.abs(hiforce - loforce)
#         rerr = aerr / np.abs(hiforce)
#         max_abs_err = np.maximum(max_abs_err, aerr)
#         max_rel_err = np.maximum(max_rel_err, rerr)
#         abs_err += weight * aerr
#         rel_err += weight * rerr

#     abs_err /= sum(w for __, w in scheme)
#     rel_err /= sum(w for __, w in scheme)
#     return np.concatenate([abs_err, rel_err, max_abs_err, max_rel_err])


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
@click.option('--period', default=10.0)
@click.option('--nsteps', default=500)
@click.option('--timestep', default=0.5)
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@util.common_args
def solve(velocity, viscosity, period, nsteps, timestep, fast, piola):
    case = get_case(fast, piola)
    mu = case.parameter(velocity=velocity, viscosity=viscosity)

    with util.time():
        solutions = solvers.navierstokes_time(case, mu, maxit=10, nsteps=nsteps, dt=timestep, solver='mkl',
                                              initsol=np.load('sln.npy'))
                                              # )

    forces = []
    for i, (mu, lhs) in enumerate(solutions[::1]):
        visualization.velocity(case, mu, lhs, name='dull', axes=False, colorbar=True, index=i, ndigits=4, streams=False)
        if 'force' in case:
            force = case['force'](mu, cont=(lhs, None))
        elif 'xforce' in case and 'yforce' in case:
            wx, wy, l = case['xforce'](mu), case['yforce'](mu), case.lift(mu)
            u = lhs + l
            xf = (
                case['divergence'](mu, cont=(wx,u)) +
                case['laplacian'](mu, cont=(wx,u)) +
                case['convection'](mu, cont=(wx,u,u))
            )
            yf = (
                case['divergence'](mu, cont=(wy,u)) +
                case['laplacian'](mu, cont=(wy,u)) +
                case['convection'](mu, cont=(wy,u,u))
            )
            xff = case['divergence'](mu, cont=(wx,u)) + case['laplacian'](mu, cont=(wx,u))
            yff = case['divergence'](mu, cont=(wy,u)) + case['laplacian'](mu, cont=(wy,u))
            force = [xf, yf, xff, yff]
        forces.append([mu['time'], *force])
        print('Step {}: time {:.2f} force {}'.format(i, mu['time'], force))

    np.save('forces.npy', np.array(forces))


# @main.command()
# @click.option('--angle', default=0.0)
# @click.option('--velocity', default=1.0)
# @click.option('--piola/--no-piola', default=False)
# @click.option('--sups/--no-sups', default=True)
# @click.option('--nred', '-r', default=10)
# @click.option('--index', '-i', default=0)
# @util.common_args
# def rsolve(angle, velocity, piola, sups, nred, index):
#     case = get_reduced(piola=piola, sups=sups, nred=nred)
#     angle = -angle / 180 * np.pi
#     mu = case.parameter(angle=angle, velocity=velocity)
#     with util.time():
#         try:
#             lhs = solvers.navierstokes_block(case, mu)
#         except AssertionError:
#             log.user('solving non-block')
#             lhs = solvers.navierstokes(case, mu)

#     visualization.velocity(case, mu, lhs, name='red', axes=False, colorbar=True)
#     visualization.pressure(case, mu, lhs, name='red', axes=False, colorbar=True)


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--num', '-n', default=8)
@util.common_args
def ensemble(fast, piola, num):
    get_ensemble(fast, piola, num)


@main.command()
@click.option('--fast/--no-fast', default=False)
@click.option('--piola/--no-piola', default=False)
@click.option('--sups/--no-sups', default=True)
@click.option('--num', '-n', default=8)
@click.option('--nred', '-r', default=10)
@util.common_args
def reduce(fast, piola, sups, num, nred):
    get_reduced(piola, sups, nred, fast, num)


# def _divs(fast: bool = False, piola: bool = False, num=8):
#     __, solutions, supremizers = get_ensemble(fast, piola, num)
#     case = get_case(fast, piola)

#     angles = np.linspace(-35, 35, 15) * np.pi / 180

#     eig_sol = reduction.eigen(case, solutions, fields=['v', 'p'])
#     rb_sol = reduction.reduced_bases(case, solutions, eig_sol, (20, 20))

#     results = []
#     for angle in angles:
#         log.user(f'angle = {angle}')

#         mu = case.parameter(angle=angle)
#         geom = case.physical_geometry(mu)

#         divs = []
#         for i in range(10):
#             vsol, = case.solution(rb_sol['v'][i], mu, ['v'], lift=True)
#             div = np.sqrt(case.domain.integrate(vsol.div(geom)**2, geometry=geom, ischeme='gauss9'))
#             log.user(f'{i}: {div:.2e}')
#             divs.append(div)

#         results.append([angle*180/np.pi, max(divs), np.mean(divs)])

#     results = np.array(results)
#     np.savetxt(util.make_filename(_divs, 'airfoil-divs-{piola}.csv', piola=piola), results)


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


def _postprocess():
    tcase = get_case(fast=True, piola=True)
    ensemble = util.filecache('alecyl-piola-sups.rens')(lambda: None)()

    res = []
    tres = [np.linspace(0, 250, num=501)]
    for nr in [10, 20, 30, 40, 50, 60, 70, 80]:
        rcase = get_reduced(piola=True, sups=True, nred=nr)
        mu = tcase.parameter()
        verrs = ensemble.errors(tcase, 'solutions', rcase, f'rsol-{nr}', tcase['v-h1s'](mu))
        perrs = ensemble.errors(tcase, 'solutions', rcase, f'rsol-{nr}', tcase['p-l2'](mu))
        res.append([rcase.size // 3, rcase.meta['err-v'], rcase.meta['err-p'], *verrs, *perrs])

        verrs = ensemble.errors(tcase, 'solutions', rcase, f'rsol-{nr}', tcase['v-h1s'](mu), summary=False)
        perrs = ensemble.errors(tcase, 'solutions', rcase, f'rsol-{nr}', tcase['p-l2'](mu), summary=False)
        abs_verrs = [
            sum(wt * ve for (wt, *__), (ve, __) in zip(ensemble.scheme[i::501], verrs[i::501])) /
            sum(wt for (wt, *__) in ensemble.scheme[i::501])
            for i in range(501)
        ]
        rel_verrs = [
            sum(wt * ve for (wt, *__), (__, ve) in zip(ensemble.scheme[i::501], verrs[i::501])) /
            sum(wt for (wt, *__) in ensemble.scheme[i::501])
            for i in range(501)
        ]
        abs_perrs = [
            sum(wt * ve for (wt, *__), (ve, __) in zip(ensemble.scheme[i::501], verrs[i::501])) /
            sum(wt for (wt, *__) in ensemble.scheme[i::501])
            for i in range(501)
        ]
        rel_perrs = [
            sum(wt * ve for (wt, *__), (__, ve) in zip(ensemble.scheme[i::501], verrs[i::501])) /
            sum(wt for (wt, *__) in ensemble.scheme[i::501])
            for i in range(501)
        ]

        tres.append(abs_verrs)
        tres.append(rel_verrs)
        tres.append(abs_perrs)
        tres.append(rel_perrs)

    # Case size, exp v, exp p,
    # mean abs v, mean rel v, max abs v, max rel v,
    # mean abs p, mean rel p, max abs p, max rel p,
    res = np.array(res)
    np.savetxt('alecyl-results-piola-sups-no-block.csv', res)
    tres = np.array(tres).T
    np.savetxt('alecyl-results-piola-sups-no-block-time.csv', tres)


@main.command()
@util.common_args
def postprocess():
    _postprocess()


@util.filecache('alecyl-piola-sups-{nred}.rcase')
def zomg(nred: int, rcase, piola: bool = True, sups: bool = True):
    rcase.bases['v'].end = nred
    rcase.bases['s'].start = nred
    rcase.bases['s'].end = 2 * nred
    rcase.bases['p'].start = 2 * nred
    rcase.bases['p'].end = 3 * nred
    rcase.bases['v'].obj = rcase.bases['v'].obj[:nred, ...]
    rcase.bases['s'].obj = rcase.bases['s'].obj[:nred, ...]
    rcase.bases['p'].obj = rcase.bases['p'].obj[:nred, ...]

    i1 = [*range(nred), *range(80, 80+nred), *range(160, 160+nred)]
    i2 = np.ix_(i1, i1)
    i3 = np.ix_(i1, i1, i1)

    rcase.projection = rcase.projection[i1,:]

    def do(itg, i):
        itg.values = [NumpyArrayIntegrand(v.obj[i]) for v in itg.values]

    do(rcase.integrals['convection'], i3)
    do(rcase.integrals['convection']._lift[frozenset((1,))], i2)
    do(rcase.integrals['convection']._lift[frozenset((2,))], i2)
    do(rcase.integrals['convection']._lift[frozenset((1,2))], i1)
    do(rcase.integrals['divergence'], i2)
    do(rcase.integrals['divergence']._lift[frozenset((0,))], i1)
    do(rcase.integrals['forcing'], i1)
    do(rcase.integrals['laplacian'], i2)
    do(rcase.integrals['laplacian']._lift[frozenset((1,))], i1)
    do(rcase.integrals['mass-lift-dt'], i1)
    do(rcase.integrals['p-l2'], i2)
    do(rcase.integrals['p-l2']._lift[frozenset((1,))], i1)
    do(rcase.integrals['v-h1s'], i2)
    do(rcase.integrals['v-h1s']._lift[frozenset((1,))], i1)
    do(rcase.integrals['v-l2'], i2)
    do(rcase.integrals['v-l2']._lift[frozenset((1,))], i1)

    alleigs_v = np.load('alleigs-v.npy')
    alleigs_s = np.load('alleigs-s.npy')
    alleigs_p = np.load('alleigs-p.npy')

    rcase.meta['err-v'] = np.sqrt(1.0 - np.sum(alleigs_v[:nred]) / np.sum(alleigs_v))
    rcase.meta['err-s'] = np.sqrt(1.0 - np.sum(alleigs_s[:nred]) / np.sum(alleigs_s))
    rcase.meta['err-p'] = np.sqrt(1.0 - np.sum(alleigs_p[:nred]) / np.sum(alleigs_p))

    rcase._cons = rcase._cons[:3*nred]

    return rcase


if __name__ == '__main__':
    main()

    # _postprocess()
    # _bfuns()

    # for k in [10, 20, 30, 40, 50, 60, 70]:
    #     rcase = get_reduced(nred=80, piola=True, sups=True)
    #     zomg(k, rcase)
