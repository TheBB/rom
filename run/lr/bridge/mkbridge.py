from splipy import (
    curve_factory as cf,
    surface_factory as sf,
    volume_factory as vf,
    io, SplineObject, SplineModel, BSplineBasis
)
from splipy.state import state
from itertools import chain
from functools import lru_cache, partial
from collections import namedtuple
import numpy as np
from math import ceil


def lru_prop(func):
    return property(lru_cache(None)(func))


def _iter_patches(data):
    for d in data:
        if isinstance(d, SplineObject):
            yield d
        else:
            yield from _iter_patches(d)


def _map_patches(func, data):
    if isinstance(data, SplineObject):
        return func(data)
    elif isinstance(data, list):
        return [_map_patches(func, d) for d in data]
    else:
        return type(data)(*(_map_patches(func, d) for d in data))


def _flipx(patch):
    patch = patch.clone()
    patch.reverse('u')
    patch.controlpoints[..., 0] = -patch.controlpoints[..., 0]
    return patch


def _translate(diff):
    return lambda patch: patch + diff


def _raise_order(order, patch):
    diff = tuple(order - r for r in patch.order())
    patch.raise_order(*diff)
    return patch


def _interpolate(pts, diff, order):
    assert order in (2, 3, 4)
    kts = np.arange(len(pts))

    if order in (3, 4):
        curve = cf.cubic_curve(pts, t=kts, boundary=cf.Boundary.TANGENT, tangents=diff)
        if order == 3:
            curve = curve.rebuild(order, len(curve) - 1)
        return curve

    basis = BSplineBasis(order, [0.0] + list(np.arange(len(pts))) + [float(len(pts) - 1)])
    return cf.interpolate(pts, basis, kts)


BridgeSupport = namedtuple('BridgeSupport', ['btm', 'top', 'walls'])
BridgeFlange = namedtuple('BridgeFlange', ['btm', 'top', 'walls', 'extra', 'fill'])
BridgeColumn = namedtuple('BridgeColumn', ['base', 'col', 'wings'])
BridgeSpan = namedtuple('BridgeSpan', ['flange_lft', 'column', 'flange_rgt'])
BridgeSupports = namedtuple('BridgeSupports', ['lft', 'rgt'])
BridgeFull = namedtuple('BridgeFull', ['span_lft', 'span_rgt', 'supports'])


class BridgeSpec:

    column_thickness = 1.0
    column_width = 7.0
    column_height = 30.0
    column_base_thickness = 3.5
    column_base_width = 7.4
    column_base_height = 2.0

    flange_height_max = 0.56
    flange_height_min = 0.2
    flange_width = 5.0

    crossec_wall_thickness = 0.2
    road_thickness = 0.23
    road_extra_thickness = 0.33
    road_extra_width = 6.0
    road_width = 8.0

    crossec_height_max = 7.0
    crossec_height_min = 2.0
    span_length = 110.0
    alpha = 2.0

    resolution = 0.5
    flange_nel = 3
    colt_nel = 1
    colw_nel = 3
    colh_nel = 11

    order = 4

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        assert self.flange_width < self.column_width
        assert self.column_thickness < self.column_base_thickness
        assert self.column_width < self.column_base_width
        assert self.column_width < self.road_width


class Bridge:

    def __init__(self, spec: BridgeSpec):
        self.spec = spec
        _map_patches(partial(_raise_order, spec.order), self.full)

    def curve(self, hmin, hmax, x, diff=False):
        if diff:
            c = (hmax - hmin) * (1 - 2 * x / self.spec.span_length) ** (self.spec.alpha - 1)
            return - c * 2 / self.spec.span_length * self.spec.alpha
        else:
            return hmin + (hmax - hmin) * (1 - 2 * x / self.spec.span_length) ** self.spec.alpha

    def csec_lo(self, x, diff=False):
        c = self.curve(self.spec.crossec_height_min, self.spec.crossec_height_max, x, diff=diff)
        if diff:
            return -c
        else:
            return self.spec.column_height - c

    def csec_hi(self, x, **kwargs):
        c = self.csec_lo(x, **kwargs)
        c += self.curve(self.spec.flange_height_min, self.spec.flange_height_max, x, **kwargs)
        return c

    def flange_curve(self, xpts, z):
        kts = np.arange(len(xpts))
        zpts = z(xpts) if callable(z) else z * np.ones(xpts.shape)
        diff = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        if callable(z):
            diff[0,2] = z(xpts[0], diff=True)
            diff[-1,2] = z(xpts[-1], diff=True)
        curve = _interpolate(np.array([xpts, np.zeros(xpts.shape), zpts]).T, diff, self.spec.order)
        return curve

    def flange_patch(self, xmin, xmax, ymin, ymax, zmin, zmax, res):
        nels = int(ceil((xmax - xmin) / res))
        xpts = np.linspace(xmin, xmax, nels + 1)

        btm = sf.edge_curves(
            self.flange_curve(xpts, zmin) + (0, ymin, 0),
            self.flange_curve(xpts, zmin) + (0, ymax, 0),
        )

        top = sf.edge_curves(
            self.flange_curve(xpts, zmax) + (0, ymin, 0),
            self.flange_curve(xpts, zmax) + (0, ymax, 0),
        )

        return _raise_order(self.spec.order, vf.edge_surfaces(btm, top))

    @lru_prop
    def flange_xpts(self):
        return np.array([
            self.spec.column_thickness,
            self.spec.column_thickness + self.spec.road_extra_width * 2,
            self.spec.span_length,
        ]) / 2

    @lru_prop
    def flange_ypts(self):
        return np.array([
            -self.spec.flange_width,
            -self.spec.flange_width + self.spec.crossec_wall_thickness * 2,
            self.spec.flange_width - self.spec.crossec_wall_thickness * 2,
            self.spec.flange_width,
        ]) / 2

    @lru_prop
    def support_xpts(self):
        return np.array([
            -self.spec.span_length/2 - 2,
            -self.spec.span_length/2,
        ])

    @lru_prop
    def col_base_surface(self):
        xpts = np.array([
            -self.spec.column_base_thickness,
            -self.spec.column_thickness,
            self.spec.column_thickness,
            self.spec.column_base_thickness,
        ]) / 2

        ypts = np.array([
            -self.spec.column_base_width,
            -self.spec.column_width,
            -self.spec.flange_width,
            -self.spec.flange_width + self.spec.crossec_wall_thickness * 2,
            self.spec.flange_width - self.spec.crossec_wall_thickness * 2,
            self.spec.flange_width,
            self.spec.column_width,
            self.spec.column_base_width,
        ]) / 2

        temp = _map_patches(partial(_raise_order, self.spec.order), [[
            sf.edge_curves(
                cf.line((xl, yl, 0), (xh, yl, 0)),
                cf.line((xl, yh, 0), (xh, yh, 0)),
            )
            for yl, yh in zip(ypts, ypts[1:])
        ] for xl, xh in zip(xpts, xpts[1:])])

        for patch in _iter_patches(temp):
            patch.refine(self.spec.colt_nel - 1, direction='u')
        for patchlist in temp:
            patchlist[3].refine(self.spec.colw_nel - 1, direction='v')
        return temp

    @lru_prop
    def col_base(self):
        return [[
            vf.extrude(patch, (0, 0, self.spec.column_base_height))
            for patch in patches
        ] for patches in self.col_base_surface]

    @lru_prop
    def col(self):
        zpts = np.array([
            self.spec.column_base_height,
            self.csec_lo(self.spec.column_thickness/2),
            self.csec_hi(self.spec.column_thickness/2),
            self.spec.column_height - self.spec.road_extra_thickness,
            self.spec.column_height - self.spec.road_thickness,
            self.spec.column_height,
        ])

        temp = _map_patches(partial(_raise_order, self.spec.order), [
            [vf.extrude(patch + (0, 0, zl), (0, 0, zh - zl)) for zl, zh in zip(zpts, zpts[1:])]
            for patch in self.col_base_surface[1][1:-1]
        ])
        for z in [temp[0], temp[-1]]:
            del z[-3:]
        for z in temp[1:-1]:
            z[-3].refine(self.spec.flange_nel - 1, direction='w')
        for z in temp:
            z[0].refine(self.spec.colh_nel, direction='w')
        return temp

    @lru_prop
    def flange_btm(self):
        xpts = self.flange_xpts
        ypts = self.flange_ypts

        temp = [[
            self.flange_patch(xl, xh, yl, yh, self.csec_lo, self.csec_hi, self.spec.resolution)
            for yl, yh in zip(ypts, ypts[1:])
        ] for xl, xh in zip(xpts, xpts[1:])]

        for patchlist in temp:
            patchlist[1].refine(2, direction='v')
        return temp

    @lru_prop
    def flange_top(self):
        xpts = self.flange_xpts
        ypts = self.flange_ypts
        zmin = self.spec.column_height - self.spec.road_thickness
        zmax = self.spec.column_height

        temp = [[
            self.flange_patch(xl, xh, yl, yh, zmin, zmax, self.spec.resolution)
            for yl, yh in zip(ypts, ypts[1:])
        ] for xl, xh in zip(xpts, xpts[1:])]

        for patchlist in temp:
            patchlist[1].refine(self.spec.flange_nel - 1, direction='v')
        return temp

    @lru_prop
    def flange_wall(self):
        xpts = self.flange_xpts
        ypts = [self.flange_ypts[:2], self.flange_ypts[-2:]]
        zpts = [
            self.csec_hi,
            self.spec.column_height - self.spec.road_extra_thickness,
            self.spec.column_height - self.spec.road_thickness,
        ]

        temp = [[
            [
                self.flange_patch(xl, xh, yl, yh, zmin, zmax, self.spec.resolution)
                for zmin, zmax in zip(zpts, zpts[1:])
            ] for yl, yh in ypts
        ] for xl, xh in zip(xpts, xpts[1:])]

        for patchlist in chain(*temp):
            patchlist[0].refine(self.spec.flange_nel - 1, direction='w')

        return temp

    @lru_prop
    def flange_extra(self):
        patch = self.flange_patch(
            *self.flange_xpts[:2],
            *self.flange_ypts[1:3],
            self.spec.column_height - self.spec.road_extra_thickness,
            self.spec.column_height - self.spec.road_thickness,
            self.spec.resolution,
        )

        patch.refine(2, direction='v')
        return patch

    @lru_prop
    def road_fill(self):
        xpts = self.flange_xpts
        ypts = [
            (-self.spec.road_width/2, -self.spec.flange_width/2),
            (self.spec.flange_width/2, self.spec.road_width/2),
        ]
        zmin = self.spec.column_height - self.spec.road_thickness
        zmax = self.spec.column_height

        return [[
            self.flange_patch(xl, xh, yl, yh, zmin, zmax, self.spec.resolution)
            for yl, yh in ypts
        ] for xl, xh in zip(xpts, xpts[1:])]

    @lru_prop
    def road_wings(self):
        xmin = -self.spec.column_thickness / 2
        xmax = self.spec.column_thickness/2
        ypts = [
            (-self.spec.road_width/2, -self.spec.flange_width/2),
            (self.spec.flange_width/2, self.spec.road_width/2),
        ]
        zmin = self.spec.column_height - self.spec.road_thickness
        zmax = self.spec.column_height

        temp = [
            vf.extrude(
                sf.extrude(
                    cf.line((xmin, ymin, zmin), (xmax, ymin, zmin)),
                    (0, ymax - ymin, 0)
                ),
                (0, 0, zmax - zmin)
            )
            for ymin, ymax in ypts
        ]

        for patch in temp:
            patch.refine(self.spec.colt_nel - 1, direction='u')
        return temp

    @lru_prop
    def support_btm(self):
        xmin, xmax = self.support_xpts
        ypts = self.flange_ypts
        zmin = self.spec.column_height - self.spec.crossec_height_min
        zmax = self.spec.column_height - self.spec.crossec_height_min + self.spec.flange_height_min

        temp = _map_patches(partial(_raise_order, self.spec.order), [
            self.flange_patch(xmin, xmax, yl, yh, zmin, zmax, self.spec.resolution)
            for yl, yh in zip(ypts, ypts[1:])
        ])

        temp[1].refine(2, direction='v')
        return temp

    @lru_prop
    def support_top(self):
        xmin, xmax = self.support_xpts
        ypts = self.flange_ypts
        zmin = self.spec.column_height - self.spec.road_thickness
        zmax = self.spec.column_height

        temp = [
            self.flange_patch(xmin, xmax, yl, yh, zmin, zmax, self.spec.resolution)
            for yl, yh in zip(ypts, ypts[1:])
        ]

        temp[1].refine(2, direction='v')
        return temp

    @lru_prop
    def support_wall(self):
        xmin, xmax = self.support_xpts
        ypts = [self.flange_ypts[:2], self.flange_ypts[-2:]]
        zpts = [
            self.spec.column_height - self.spec.crossec_height_min + self.spec.flange_height_min,
            self.spec.column_height - self.spec.road_extra_thickness,
            self.spec.column_height - self.spec.road_thickness,
        ]

        temp = _map_patches(partial(_raise_order, self.spec.order), [[
            self.flange_patch(xmin, xmax, yl, yh, zmin, zmax, self.spec.resolution)
            for zmin, zmax in zip(zpts, zpts[1:])
        ] for yl, yh in ypts])

        for patchlist in temp:
            patchlist[0].refine(2, direction='w')
        return temp

    @lru_prop
    def support(self):
        return BridgeSupport(self.support_btm, self.support_top, self.support_wall)

    @lru_prop
    def flange(self):
        return BridgeFlange(
            self.flange_btm,
            self.flange_top,
            self.flange_wall,
            self.flange_extra,
            self.road_fill,
        )

    @lru_prop
    def column(self):
        return BridgeColumn(
            self.col_base,
            self.col,
            self.road_wings,
        )

    @lru_prop
    def span(self):
        return BridgeSpan(
            _map_patches(_flipx, self.flange),
            self.column,
            self.flange,
        )

    @lru_prop
    def supports(self):
        return BridgeSupports(
            self.support,
            _map_patches(_translate((self.spec.span_length, 0, 0)), _map_patches(_flipx, self.support)),
        )

    @lru_prop
    def full(self):
        return BridgeFull(
            self.span,
            _map_patches(_translate((self.spec.span_length, 0, 0)), self.span),
            self.supports,
        )

    def patches(self):
        yield from _iter_patches(_map_patches(_translate((-self.spec.span_length/2, 0, 0)), self.full))


def main():
    spec = BridgeSpec(order=2, resolution=2.0)
    bridge = Bridge(spec)
    patches = list(bridge.patches())

    for p in patches:
        assert p.order() == (spec.order,) * 3

    with io.G2('bridge.g2') as g2:
        g2.write(patches)

    model = SplineModel()
    model.add(patches)

    for patch in _iter_patches(bridge.full.span_lft.column.base):
        model[patch.section(w=0)].name = 'fundament'
    for patch in _iter_patches(bridge.full.span_rgt.column.base):
        model[patch.section(w=0)].name = 'fundament'
    for patch in _iter_patches(bridge.full.supports.lft.btm):
        model[patch.section(w=0)].name = 'support'
    for patch in _iter_patches(bridge.full.supports.rgt.btm):
        model[patch.section(w=0)].name = 'support'
    for patch in _iter_patches(bridge.full.span_lft.flange_lft.top):
        model[patch.section(w=-1)].name = 'road'
    for patch in _iter_patches(bridge.full.span_lft.flange_rgt.top):
        model[patch.section(w=-1)].name = 'road'
    for patch in _iter_patches(bridge.full.span_rgt.flange_lft.top):
        model[patch.section(w=-1)].name = 'road'
    for patch in _iter_patches(bridge.full.span_rgt.flange_rgt.top):
        model[patch.section(w=-1)].name = 'road'
    for patchlist in bridge.full.span_lft.column.col[1:-1]:
        model[patchlist[-1].section(w=-1)].name = 'road'
    for patchlist in bridge.full.span_rgt.column.col[1:-1]:
        model[patchlist[-1].section(w=-1)].name = 'road'
    for patch in _iter_patches(bridge.full.span_lft.flange_lft.fill):
        model[patch.section(w=-1)].name = 'road'
    for patch in _iter_patches(bridge.full.span_lft.flange_rgt.fill):
        model[patch.section(w=-1)].name = 'road'
    for patch in _iter_patches(bridge.full.span_rgt.flange_lft.fill):
        model[patch.section(w=-1)].name = 'road'
    for patch in _iter_patches(bridge.full.span_rgt.flange_rgt.fill):
        model[patch.section(w=-1)].name = 'road'
    for patch in bridge.full.span_lft.column.wings:
        model[patch.section(w=-1)].name = 'road'
    for patch in bridge.full.span_rgt.column.wings:
        model[patch.section(w=-1)].name = 'road'

    model.summary()
    model.write_ifem('bridge')


if __name__ == '__main__':
    main()
