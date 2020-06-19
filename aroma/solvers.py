# Copyright (C) 2014 SINTEF ICT,
# Applied Mathematics, Norway.
#
# Contact information:
# E-mail: eivind.fonn@sintef.no
# SINTEF Digital, Department of Applied Mathematics,
# P.O. Box 4760 Sluppen,
# 7045 Trondheim, Norway.
#
# This file is part of AROMA.
#
# AROMA is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AROMA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with AROMA. If not, see
# <http://www.gnu.org/licenses/>.
#
# In accordance with Section 7(b) of the GNU General Public License, a
# covered work must retain the producer line in every data file that
# is created or manipulated using AROMA.
#
# Other Usage
# You can be released from the requirements of the license by purchasing
# a commercial license. Buying such a license is mandatory as soon as you
# develop commercial activities involving the AROMA library without
# disclosing the source code of your own applications.
#
# This file may be used in accordance with the terms contained in a
# written agreement between you and SINTEF Digital.


from itertools import count
import numpy as np
import scipy as sp
from nutils import function as fn, log, matrix

from aroma.affine import integrate


class IterationCountError(Exception):
    pass


def solve(mx, rhs, cons, solver='spsolve', **kwargs):
    if solver == 'mkl':
        if isinstance(mx, np.ndarray):
            raise TypeError
        mx = sp.sparse.coo_matrix(mx)
        mx = matrix.MKLMatrix(mx.data, np.array([mx.row, mx.col]), mx.shape)
        return mx.solve(rhs, constrain=cons, **kwargs)

    elif isinstance(mx, np.matrix):
        mx = matrix.NumpyMatrix(np.array(mx))
        return mx.solve(rhs, constrain=cons, **kwargs)

    elif isinstance(mx, np.ndarray):
        mx = matrix.NumpyMatrix(mx)
        return mx.solve(rhs, constrain=cons, **kwargs)

    else:
        mx = matrix.ScipyMatrix(mx)
        return mx.solve(rhs, constrain=cons, solver=solver, **kwargs)


def _stokes_matrix(case, mu, div=True, **kwargs):
    matrix = case['laplacian'](mu)
    if div:
        matrix += case['divergence'](mu, sym=True)
    if 'stab-lhs' in case:
        matrix += case['stab-lhs'](mu, sym=True)
    return matrix


def _stokes_rhs(case, mu, **kwargs):
    rhs = - case['divergence'](mu, cont=('lift', None)) - case['laplacian'](mu, cont=(None, 'lift'))
    if 'forcing' in case:
        rhs += case['forcing'](mu)
    if 'stab-lhs' in case:
        rhs -= case['stab-lhs'](mu, cont=(None, 'lift'))
    if 'stab-rhs' in case:
        rhs += case['stab-rhs'](mu)
    return rhs


def _stokes_assemble(case, mu, **kwargs):
    return _stokes_matrix(case, mu, **kwargs), _stokes_rhs(case, mu, **kwargs)


def stokes(case, mu):
    assert 'divergence' in case
    assert 'laplacian' in case

    matrix, rhs = _stokes_assemble(case, mu)
    lhs = solve(matrix, rhs, case.constraints)

    return lhs


def navierstokes_conv(case, mu, lhs):
    c = case['convection']
    rh = c(mu, cont=(None, lhs, lhs))
    lh = c(mu, cont=(None, lhs, None)) + c(mu, cont=(None, None, lhs))
    rh, lh = integrate(rh, lh)
    return rh, lh


def navierstokes(case, mu, newton_tol=1e-10, maxit=10, **kwargs):
    assert 'divergence' in case
    assert 'laplacian' in case
    assert 'convection' in case
    assert 'v-h1s' in case

    stokes_mat, stokes_rhs = _stokes_assemble(case, mu)
    lhs = solve(stokes_mat, stokes_rhs, case.constraints)

    lift = case['lift'](mu)
    stokes_mat += case['convection'](mu, cont=(None, 'lift', None)) + case['convection'](mu, cont=(None, None, 'lift'))
    stokes_rhs -= case['convection'](mu, cont=(None, 'lift', 'lift'))

    vmass = case['v-h1s'](mu)

    for it in count(1):
        rh, lh = navierstokes_conv(case, mu, lhs)
        rhs = stokes_rhs - stokes_mat @ lhs - rh
        ns_mat = stokes_mat + lh

        update = solve(ns_mat, rhs, case.constraints, **kwargs)
        lhs += update

        update_norm = np.sqrt(update @ vmass @ update)
        log.user('update: {:.2e}'.format(update_norm))
        if update_norm < newton_tol:
            break

        if it > maxit:
            raise IterationCountError

    return lhs


def navierstokes_timestep(case, mu, dt, cursol, newton_tol=1e-10, maxit=10, tsolver='be', **kwargs):
    assert tsolver in ('be', 'cn')

    stokes_mat, stokes_rhs = _stokes_assemble(case, mu, div=(tsolver == 'be'))
    stokes_mat += case['convection'](mu, lift=1) + case['convection'](mu, lift=2)
    stokes_rhs -= case['convection'](mu, lift=(1,2))

    if tsolver == 'cn':
        # TODO: The computations here should be computed with the previous timestep's mu
        # this technically assumes that the problem has no explicit time dependency
        stokes_mat /= 2
        stokes_rhs -= stokes_mat @ cursol
        rh_cn = case['convection'](mu, cont=(None, cursol, cursol))
        rh_cn, = integrate(rh_cn)
        stokes_rhs -= rh_cn / 2

    sys_mat = stokes_mat + case['v-l2'](mu) / dt
    if tsolver == 'cn':
        divmx = case['divergence'](mu, sym=True)
        sys_mat += divmx

    vmass_h1 = case['v-h1s'](mu)
    pmass_l2 = case['v-h1s'](mu)
    vmass_l2 = case['v-l2'](mu)

    if 'mass-lift-dt' in case:
        stokes_rhs -= case['mass-lift-dt'](mu)

    lhs = np.copy(cursol)
    for it in count(1):
        rh, lh = navierstokes_conv(case, mu, lhs)

        if tsolver == 'cn':
            rh /= 2
            lh /= 2

        rhs = stokes_rhs - stokes_mat @ lhs - rh
        ns_mat = sys_mat + lh

        if tsolver == 'cn':
            rhs -= divmx @ lhs

        update = solve(ns_mat, rhs, case.constraints, **kwargs)
        lhs += update
        stokes_rhs -= vmass_l2 @ update / dt

        update_norm = np.sqrt(update @ vmass_h1 @ update)
        update_norm += np.sqrt(update @ pmass_l2 @ update)
        log.user('update: {:.2e}'.format(update_norm))
        if update_norm < newton_tol:
            break

        if it > maxit:
            raise IterationCountError

    return lhs


def navierstokes_time(case, mu, dt=1e-2, nsteps=100, timename='time', initsol=None, **kwargs):
    assert 'divergence' in case
    assert 'laplacian' in case
    assert 'convection' in case
    assert 'v-h1s' in case
    assert 'v-l2' in case

    if initsol is None:
        stokes_mat, stokes_rhs = _stokes_assemble(case, mu)
        lhs = solve(stokes_mat, stokes_rhs, case.constraints)
    else:
        lhs = initsol
    vmass_h1 = case['v-h1s'](mu)

    yield (mu, lhs)
    for istep in range(1, nsteps+1):
        mu = dict(**mu)
        mu[timename] += dt
        with log.context(f'Step {istep} (t = {mu[timename]:.2f})'):
            lhs = navierstokes_timestep(case, mu, dt, lhs, **kwargs)
        yield(mu, lhs)


def blocksolve_velocity(vv, rhs, V):
    lhs = np.zeros_like(rhs)
    lhs[V] = matrix.NumpyMatrix(vv).solve(rhs[V])
    return lhs


def blocksolve_pressure(sp, rhs, S, P):
    lhs = np.zeros_like(rhs)
    lhs[P] = matrix.NumpyMatrix(sp).solve(rhs[S])
    return lhs


def navierstokes_block(case, mu, newton_tol=1e-10, maxit=10):
    for itg in ['laplacian-vv', 'laplacian-sv', 'divergence-sp',
                'convection-vvv', 'convection-svv']:
        assert itg in case

    nn = case.ndofs // 3
    V = np.arange(nn)
    S = np.arange(nn,2*nn)
    P = np.arange(2*nn,3*nn)

    # Assumption: divergence of lift is zero
    stokes_rhs = np.zeros((case.ndofs,))
    stokes_rhs[V] -= case['laplacian-vv'](mu, cont=(None, 'lift'))
    stokes_rhs[S] -= case['laplacian-sv'](mu, cont=(None, 'lift'))

    mvv = case['laplacian-vv'](mu)
    msv = case['laplacian-sv'](mu)
    msp = case['divergence-sp'](mu)
    lhs = blocksolve_velocity(mvv, stokes_rhs, V)

    mvv += (case['convection-vvv'](mu, cont=(None, 'lift', None)) +
            case['convection-vvv'](mu, cont=(None, None, 'lift')))
    msv += (case['convection-svv'](mu, cont=(None, 'lift', None)) +
            case['convection-svv'](mu, cont=(None, None, 'lift')))

    stokes_rhs[V] -= case['convection-vvv'](mu, cont=(None, 'lift', 'lift'))
    stokes_rhs[S] -= case['convection-svv'](mu, cont=(None, 'lift', 'lift'))

    vmass = case['v-h1s'](mu)

    for it in count(1):
        cc = case['convection-vvv']
        nvv = mvv + cc(mu, cont=(None, lhs[V], None)) + cc(mu, cont=(None, None, lhs[V]))

        rhs = stokes_rhs.copy()
        rhs[V] -= mvv @ lhs[V]
        rhs[V] -= case['convection-vvv'](mu, cont=(None, lhs[V], lhs[V]))

        update = blocksolve_velocity(nvv, rhs, V)
        lhs += update

        update_norm = np.sqrt(update @ vmass @ update)
        log.user('update: {:.2e}'.format(update_norm))
        if update_norm < newton_tol:
            break

        if it > maxit:
            raise IterationCountError

    rhs = stokes_rhs.copy()
    rhs[S] -= msv @ lhs[V]
    rhs[S] -= msp @ lhs[P]
    rhs[S] -= case['convection-svv'](mu, cont=(None, lhs[V], lhs[V]))
    update = blocksolve_pressure(msp, rhs, S, P)
    lhs += update

    return lhs


def supremizer(case, mu, rhs):
    conses = case.constraints
    mask = np.ones(conses.shape, dtype=np.bool)
    mask[case.bases['v'].indices] = False
    conses[mask] = 0.0

    rhs = case['divergence'](mu).dot(rhs)
    mx = case['v-h1s'](mu)
    return solve(mx, rhs, conses)


def infsup(case, mu):
    if 's' in case.bases:
        vinds = np.concatenate([case.bases['v'].indices, case.bases['s'].indices])
    else:
        vinds = np.concatenate([case.bases['v'].indices])
    pinds = case.bases['p'].indices

    vmass = case['v-h1s'](mu)[np.ix_(vinds, vinds)]
    pmass = case['p-l2'](mu)[np.ix_(pinds, pinds)]
    bmx = case['divergence'](mu)[np.ix_(vinds, pinds)]

    left = bmx.T @ np.linalg.inv(vmass) @ bmx
    return np.sqrt(sp.linalg.eigvalsh(left, pmass)[0])


def elasticity(case, mu):
    matrix = case['stiffness'](mu)
    if 'penalty' in case:
        matrix += case['penalty'](mu)
    rhs = - case['stiffness'](mu, cont=(None, 'lift'))
    if 'forcing' in case:
        rhs += case['forcing'](mu)
    if 'gravity' in case:
        rhs += case['gravity'](mu)

    try:
        # lhs = solve(matrix, rhs, case.constraints, solver='spsolve', atol=1e-10, precon='SPLU')
        lhs = solve(matrix, rhs, case.constraints, solver='mkl')
    except TypeError:
        lhs = solve(matrix, rhs, case.constraints)

    return lhs


def poisson(case, mu):
    matrix = case['laplacian'](mu)
    rhs = case['forcing'](mu) - case['laplacian'](mu, cont=(None, 'lift'))
    conses = case.constraints
    return solve(matrix, rhs, conses)


# TODO: Assumes nutils
def error(case, mu, field, lhs, norm='l2'):
    exact = case.exact(mu, field)
    numeric = case.basis(field, mu).dot(lhs)
    geom = case.geometry(mu)

    diff = exact - numeric
    if norm == 'h1':
        diff = diff.grad(geom)
    diff = fn.sum(diff * diff)

    return np.sqrt(case.domain.integrate(diff, geometry=geom, ischeme='gauss9'))


def force(case, mu, lhs, prev=None, dt=None, method='recover', tsolver='be'):
    mu = case.parameter()
    if method == 'direct':
        return case['force'](mu, cont=(lhs, None))
    assert method == 'recover'
    wx, wy = case['xforce'](mu), case['yforce'](mu)

    if tsolver == 'cn':
        assert prev is not None
        assert dt is not None
        def getf(w):
            f = (case['laplacian'](mu, cont=(w, lhs)) + integrate(case['convection'](mu, cont=(w, lhs, lhs)))) / 2
            f += (case['laplacian'](mu, cont=(w, prev)) + integrate(case['convection'](mu, cont=(w, prev, prev)))) / 2
            f += case['divergence'](mu, cont=(w, lhs))
            f += (case['v-l2'](mu, cont=(w, lhs)) - case['v-l2'](mu, cont=(w, prev))) / dt
            return f
    else:
        def getf(w):
            f = case['laplacian'](mu, cont=(w, lhs))
            f += integrate(case['convection'](mu, cont=(w, lhs, lhs)))
            f += case['divergence'](mu, cont=(w, lhs))
            if prev is not None:
                f += (case['v-l2'](mu, cont=(w, lhs)) - case['v-l2'](mu, cont=(w, prev))) / dt
            return f

    fx, fy = getf(wx), getf(wy)

    return np.array([*fx, *fy])
