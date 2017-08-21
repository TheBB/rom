import click
from collections import namedtuple
from functools import wraps
from itertools import count, product, repeat
import numpy as np
from nutils import plot, log, function as fn, _, core
from operator import itemgetter
from os.path import isfile, splitext
import pickle

import bbflow.cases as cases
from bbflow.cases.bases import Case
from bbflow.ensemble import make_ensemble, errors as ensemble_errors
import bbflow.quadrature as quadrature
import bbflow.reduction as reduction
import bbflow.solvers as solvers


def parse_extra_args(final_args=[]):
    def decorator(func):
        @wraps(func)
        def inner(ctx, **kwargs):
            extra_args = {'mu': ()}
            args = ctx.args
            if final_args:
                for (name, type_), arg in zip(final_args, args[-len(final_args):]):
                    extra_args[name] = type_.convert(arg, None, ctx)
                args = args[:-len(final_args)]
            while args:
                key, args = args[0], args[1:]
                key = key[2:].replace('-', '_')
                values = ()
                while args and not args[0].startswith('--'):
                    value, args = args[0], args[1:]
                    for cons in [int, float]:
                        try:
                            value = cons(value)
                            break
                        except ValueError: pass
                    values += (value,)
                if len(values) == 0:
                    if key.startswith('no_'):
                        extra_args[key[3:]] = False
                    else:
                        extra_args[key] = True
                elif len(values) == 1:
                    extra_args[key] = values[0]
                else:
                    extra_args[key] = values
            return func(ctx, **kwargs, **extra_args)
        return inner
    return decorator


class CaseType(click.ParamType):
    name = 'case'

    def convert(self, value, param, ctx):
        if isinstance(value, Case):
            return value
        elif value in cases.__dict__:
            return getattr(cases, value)
        elif isfile(value):
            with open(value, 'rb') as f:
                case = pickle.load(f)
            return lambda *args, **kwargs: case
        self.fail('Unknown case: {}'.format(value))


class SolverType(click.ParamType):
    name = 'solver'

    def convert(self, value, param, ctx):
        if callable(value):
            return value
        elif value in {'stokes', 'navierstokes'}:
            return getattr(solvers, value)
        self.fail('Unknown solver: {}'.format(value))


class PickledType(click.ParamType):
    name = 'object'

    def convert(self, value, param, ctx):
        if not isinstance(value, str):
            return value
        elif isfile(value):
            with open(value, 'rb') as f:
                return pickle.load(f)
        self.fail('Unknown object: {}'.format(value))


@click.group()
@click.pass_context
@click.option('--case', '-c', type=CaseType(), required=False)
@click.option('--solver', '-s', type=click.Choice(solvers.__all__), required=False)
def main(ctx, case, solver):
    pass


def command(name=None):
    def decorator(func):
        func = click.pass_context(func)
        func = main.command(
            name,
            context_settings=dict(
                ignore_unknown_options=True,
                allow_extra_args=True,
            )
        )(func)
        return func
    return decorator


@command()
@click.option('--case', '-c', type=CaseType(), required=True)
@click.option('--solver', '-s', type=SolverType(), required=True)
@parse_extra_args()
def single(ctx, case, solver, mu, **kwargs):
    case = case(**kwargs)
    mu = case.parameter(*mu)
    lhs = solver(case, mu, **kwargs)
    solvers.metrics(case, mu, lhs, **kwargs)
    solvers.plots(case, mu, lhs, fields=['v', 'p', 'vp'], **kwargs)


@command()
@click.option('--imethod', type=click.Choice(['full', 'sparse', 'uniform']), default='full')
@click.option('--ipts', type=int, default=4)
@click.option('--weights/--no-weights', default=False)
@parse_extra_args([
    ('solver', SolverType()),
    ('case', CaseType()),
    ('out', click.File(mode='wb', lazy=True)),
])
def ensemble(ctx, case, solver, imethod, out, ipts=None, weights=False, **kwargs):
    case = case(**kwargs)
    scheme = list(getattr(quadrature, imethod)(case.ranges(), ipts))
    ensemble = make_ensemble(case, solver, scheme, weights=weights)
    pickle.dump({'case': case, 'scheme': scheme, 'ensemble': ensemble}, out)


@command()
@parse_extra_args([('ensemble', PickledType())])
def spectrum(ctx, ensemble, **kwargs):
    case = ensemble['case']
    ens = ensemble['ensemble']
    decomp = reduction.eigen(case, ens, **kwargs)
    reduction.plot_spectrum(decomp, **kwargs)


@command()
@parse_extra_args([
    ('ensemble', PickledType()),
    ('out', click.File(mode='wb', lazy=True)),
])
def reduce(ctx, ensemble, out, **kwargs):
    case = ensemble['case']
    ens = ensemble['ensemble']
    decomp = reduction.eigen(case, ens, **kwargs)

    projcase = reduction.make_reduced(case, ens, decomp, **kwargs)
    pickle.dump(projcase, out)


@command('reduce-many')
@parse_extra_args([
    ('ensemble', PickledType()),
    ('out', click.STRING),
])
def reduce_many(ctx, ensemble, out, **kwargs):
    case = ensemble['case']
    ens = ensemble['ensemble']
    decomp = reduction.eigen(case, ens, **kwargs)

    nmodes = list(reduction.nmodes(decomp, **kwargs))
    nmodes = [nm for nm, __ in nmodes]
    for i, nm in enumerate(nmodes):
        print('{:04d}: {}'.format(i, nm))

    cases = reduction.make_reduced_parallel(case, ens, decomp, nmodes)
    for i, projcase in enumerate(cases):
        fn, ext = splitext(out)
        filename = '%s-%04d%s' % (fn, i, ext)
        with open(filename, 'wb') as f:
            pickle.dump(projcase, f)


@command()
@parse_extra_args([
    ('ensemble', PickledType()),
    ('solver', SolverType()),
    ('case', CaseType()),
])
def errors(ctx, ensemble, solver, case, **kwargs):
    case = case(**kwargs)
    hicase = ensemble['case']
    hifi = ensemble['ensemble']
    scheme = ensemble['scheme']
    lofi = make_ensemble(case, solver, scheme, weights=False)

    abs_err, rel_err = ensemble_errors(case, hicase.mass('v'), hifi, lofi, scheme)
    print('Mean absolute error: {:.2e}'.format(abs_err))
    print('Mean relative error: {:.2e}'.format(rel_err))


@command('errors-many')
@click.option('--out', type=click.File(mode='w', lazy=True), required=True)
@click.argument('ensemble', type=PickledType())
@click.argument('solver', type=SolverType())
@click.argument('cases', type=CaseType(), nargs=-1)
def errors_many(ctx, out, ensemble, solver, cases):
    hicase = ensemble['case']
    hifi = ensemble['ensemble']
    scheme = ensemble['scheme']

    data = []
    for i, case in enumerate(cases):
        case = case()
        lofi = make_ensemble(case, solver, scheme, weights=False)
        abs_err, rel_err = ensemble_errors(case, hicase.mass('v'), hifi, lofi, scheme)

        data.append((
            abs_err, rel_err,
            *case.meta['nmodes'].values(),
            *case.meta['errors'].values(),
        ))

    for i, row in enumerate(data):
        s = ' '.join(['{}'] * (1 + len(row))) + '\n'
        out.write(s.format(i, *row))


if __name__ == '__main__':
    main()
