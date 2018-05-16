import click
from nutils import plot, _, function as fn
import numpy as np

from aroma import cases, util, solvers, visualization
from aroma.cases.lshape import mksigma


def boundary_plots(domain, geom, sigma, suffix=''):
    x, y = geom[...,_]
    (sxx, sxy), (__, syy) = sigma

    pts, norm, tang = domain.boundary['patch0-bottom'].elem_eval([y, sxx, sxy], ischeme='bezier5', separate=True)
    with plot.PyPlot('left' + suffix, ndigits=0) as plt:
        plt.mesh(pts, norm)
        plt.mesh(pts, tang)
        plt.autoscale(enable=True, axis='both', tight=True)

    pts, norm, tang = domain.boundary['patch1-bottom'].elem_eval([x, syy, sxy], ischeme='bezier5', separate=True)
    with plot.PyPlot('bottom' + suffix, ndigits=0) as plt:
        plt.mesh(pts, norm)
        plt.mesh(pts, tang)
        plt.autoscale(enable=True, axis='both', tight=True)

    pts, norm, tang = domain.boundary['patch0-left'].elem_eval([x, syy, sxy], ischeme='bezier5', separate=True)
    with plot.PyPlot('top' + suffix, ndigits=0) as plt:
        plt.mesh(pts, norm)
        plt.mesh(pts, tang)
        plt.autoscale(enable=True, axis='both', tight=True)

    pts, norm, tang = domain.boundary['patch1-right'].elem_eval([y, sxx, sxy], ischeme='bezier5', separate=True)
    with plot.PyPlot('right' + suffix, ndigits=0) as plt:
        plt.mesh(pts, norm)
        plt.mesh(pts, tang)
        plt.autoscale(enable=True, axis='both', tight=True)


def verification_plots(domain, geom, sigma):
    pts = domain.elem_eval(geom, ischeme='bezier5', separate=True)
    with plot.PyPlot('domain', ndigits=0, figsize=(10,10)) as plt:
        plt.mesh(pts)
    boundary_plots(domain, geom, sigma)


def solution_plots(case, mu, lhs):
    disp = case.bases['u'].obj.dot(lhs)
    geom = case.refgeom
    E = mu['ymod']
    NU = mu['prat']
    MU = E / (1 + NU) / 2
    LAMBDA = E * NU / (1 + NU) / (1 - 2*NU)
    numsol = 2 * MU * disp.symgrad(geom) + LAMBDA * disp.div(geom) * fn.eye(disp.shape[0])
    anasol = mksigma(geom)

    numdiv = fn.norm2(numsol.div(geom))
    anadiv = fn.norm2(anasol.div(geom))

    pts, numdiv, anadiv = case.domain.elem_eval([geom, numdiv, anadiv], ischeme='bezier3', separate=True)
    with plot.PyPlot('num-div', ndigits=0, figsize=(10,10)) as plt:
        plt.mesh(pts, numdiv)
        plt.colorbar()
        plt.clim(0, 100)
    with plot.PyPlot('ana-div', ndigits=0, figsize=(10,10)) as plt:
        plt.mesh(pts, anadiv)
        plt.colorbar()
        plt.clim(0, 100)

    boundary_plots(case.domain, case.refgeom, numsol, suffix='-num')
    boundary_plots(case.domain, case.refgeom, anasol, suffix='-ana')

    (nxx, nxy), (__, nyy) = numsol
    (axx, axy), (__, ayy) = anasol
    pts, nxx, nyy, nxy, axx, ayy, axy = case.domain.elem_eval(
        [geom, nxx, nyy, nxy, axx, ayy, axy], ischeme='bezier3', separate=True
    )

    with plot.PyPlot('num-sxx', ndigits=0, figsize=(10,10)) as plt:
        plt.mesh(pts, nxx)
        plt.colorbar()
    with plot.PyPlot('ana-sxx', ndigits=0, figsize=(10,10)) as plt:
        plt.mesh(pts, axx)
        plt.colorbar()
    with plot.PyPlot('num-syy', ndigits=0, figsize=(10,10)) as plt:
        plt.mesh(pts, nyy)
        plt.colorbar()
    with plot.PyPlot('ana-syy', ndigits=0, figsize=(10,10)) as plt:
        plt.mesh(pts, ayy)
        plt.colorbar()


@util.filecache('lshape.case')
def get_case():
    case = cases.lshape(nrefs=7, factor=1.1)
    case.precompute(force=True)
    return case


@click.group()
def main():
    pass


@main.command()
@util.common_args
def disp():
    case = get_case()
    print(case)
    verification_plots(case.domain, case.refgeom, mksigma(case.refgeom))


@main.command()
@click.option('--ymod', default=1.0)
@click.option('--prat', default=0.3)
@util.common_args
def solve(ymod, prat):
    case = get_case()
    mu = case.parameter(ymod=ymod, prat=prat)
    with util.time():
        lhs = solvers.elasticity(case, mu)
    visualization.deformation(case, mu, lhs, name='deform')
    solution_plots(case, mu, lhs)


if __name__ == '__main__':
    main()
