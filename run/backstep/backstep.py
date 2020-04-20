import click
from functools import partial
from nutils import log, config
import multiprocessing
import h5py

from nutils import mesh, function as fn, _, matrix

from aroma.case import NutilsCase
from aroma.affine import MuLambda, MuConstant
import aroma.affine.integrands.nutils as ntl
from aroma import solvers, util, visualization, ensemble as ens, quadrature, reduction


def decorate_params(func):
    func = click.option('--length', default=10.0)(func)
    func = click.option('--height', default=1.0)(func)
    func = click.option('--velocity', default=1.0)(func)
    func = click.option('--viscosity', default=20.0)(func)
    return func


def decorate_case(func):
    func = click.option('--degree', default=2)(func)
    func = click.option('--refine', default=1)(func)
    return func


def decorate_ensemble(func):
    func = click.option('--num', default=15)(func)
    return func


def decorate_nred(func):
    func = click.option('--nred', default=10)(func)
    return func


def geometry(mu, L, H, refgeom):
    x, y = refgeom
    hx = fn.piecewise(x, (0,), 0, x)
    hy = fn.piecewise(y, (0,), y, 0)
    return (
        refgeom +
        (L - 1)(mu) * fn.asarray((hx, 0)) +
        (H - 1)(mu) * fn.asarray((0, hy))
    )


@util.filecache('backstep-{refine}-{degree}.case')
def get_case(refine: int, degree: int):
    nel_up = int(10 * refine)
    nel_length = int(100 * refine)

    up_edges = [(0, 1), (3, 4), (6, 7), (0, 3), (1, 4), (2, 3), (5, 6)]
    length_edges = [(2, 5), (3, 6), (4, 7)]
    all_edges = [*up_edges, *length_edges]

    domain, refgeom = mesh.multipatch(
        patches = [[(0, 1), (3, 4)], [(3, 4), (6, 7)], [(2, 3), (5, 6)]],
        nelems = {**{e: nel_up for e in up_edges}, **{e: nel_length for e in length_edges}},
        patchverts = [(-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    )

    case = NutilsCase('Backward-facing step channel', domain, refgeom, refgeom)

    NU = 1 / case.parameters.add('viscosity', 20, 50)
    L = case.parameters.add('length', 9, 12, 10)
    H = case.parameters.add('height', 0.3, 2, 1)
    V = case.parameters.add('velocity', 0.5, 1.2, 1)

    vxbasis = domain.basis('spline', degree=degree)
    vybasis = domain.basis('spline', degree=degree)
    pbasis = domain.basis('spline', degree=degree-1)

    vdofs = len(vxbasis) + len(vybasis)
    pdofs = len(pbasis)
    ndofs = vdofs + pdofs

    vxbasis, vybasis, pbasis = fn.chain([vxbasis, vybasis, pbasis])
    vbasis = vxbasis[:,_] * (1, 0) + vybasis[:,_] * (0, 1)

    case.bases.add('v', vbasis, length=vdofs)
    case.bases.add('p', pbasis, length=pdofs)

    case['geometry'] = MuLambda(
        partial(geometry, L=L, H=H, refgeom=refgeom),
        (2,), ('length', 'height'),
    )

    case.constrain(
        'v', 'patch0-bottom', 'patch0-top', 'patch0-left',
        'patch1-top', 'patch2-bottom', 'patch2-left',
    )

    case['divergence'] = ntl.NSDivergence(ndofs, 'length', 'height')
    case['convection'] = ntl.NSConvection(ndofs, 'length', 'height')
    case['laplacian'] = ntl.Laplacian(ndofs, 'v', 'length', 'height', scale=NU)
    case['v-h1s'] = ntl.Laplacian(ndofs, 'v', 'length', 'height')
    case['p-l2'] = ntl.Mass(ndofs, 'p', 'length', 'height')

    with matrix.Scipy():
        __, y = refgeom
        profile = fn.max(0, y * (1 - y))[_] * (1, 0)
        case['lift'] = MuConstant(case.project_lift(profile, 'v'), scale=V)

        mu = case.parameter()
        lhs = solvers.stokes(case, mu)
        case['lift'] = MuConstant(case.solution_vector(lhs, mu, lift=True), scale=V)

    return case


@util.filecache('backstep-{refine}-{degree}-{num}.ens')
def get_ensemble(refine: int, degree: int, num: int):
    case = get_case(refine, degree)
    scheme = quadrature.sparse(case.ranges(), num)
    ensemble = ens.Ensemble(scheme)
    ensemble.compute('solutions', case, solvers.navierstokes)
    ensemble.compute('supremizers', case, solvers.supremizer, args=[ensemble['solutions']])
    return ensemble


@util.filecache('backstep-{refine}-{degree}-{nred}.rcase')
def get_reduced(refine: int, degree: int, nred: int, num=None):
    case = get_case(refine, degree)
    ensemble = get_ensemble(refine, degree, num)

    case.integrals['divergence'].liftable = (0, 1)

    reducer = reduction.EigenReducer(case, ensemble)
    reducer.add_basis('v', parent='v', ensemble='solutions', ndofs=nred, norm='h1s')
    reducer.add_basis('s', parent='v', ensemble='supremizers', ndofs=nred, norm='h1s')
    reducer.add_basis('p', parent='p', ensemble='solutions', ndofs=nred, norm='l2')

    reducer.plot_spectra('backstep-spectrum')
    return reducer(tol=1e-6, nrules=3, overrides={
        'laplacian': {'tol': 1e-4, 'nrules': 4},
        'v-h1s': {'tol': 1e-4, 'nrules': 4},
    })


@click.group()
def main():
    pass


@main.command()
@decorate_case
@util.common_args
def disp(**kwargs):
    print(get_case(**kwargs))


@main.command()
@decorate_case
@decorate_params
@util.common_args
def solve(refine, degree, **kwargs):
    case = get_case(refine=refine, degree=degree)
    mu = case.parameter(**kwargs)
    with util.time():
        lhs = solvers.navierstokes(case, mu)
    visualization.velocity(case, mu, lhs, figsize=(15,3), name='full', spacing=0.1, density=1.0)
    visualization.pressure(case, mu, lhs, figsize=(15,3), name='full')


@main.command()
@decorate_case
@decorate_params
@decorate_nred
@util.common_args
def rsolve(refine, degree, nred, **kwargs):
    case = get_reduced(refine=refine, degree=degree, nred=nred)
    mu = case.parameter(**kwargs)
    with util.time():
        lhs = solvers.navierstokes(case, mu)
    visualization.velocity(case, mu, lhs, figsize=(15,3), name='red', spacing=0.1, density=1.0)
    visualization.pressure(case, mu, lhs, figsize=(15,3), name='red')


@main.command()
@decorate_case
@decorate_ensemble
@util.common_args
def ensemble(**kwargs):
    get_ensemble(**kwargs)


@main.command()
@decorate_case
@decorate_ensemble
@decorate_nred
@util.common_args
def reduce(**kwargs):
    get_reduced(**kwargs)


if __name__ == '__main__':
    main()
