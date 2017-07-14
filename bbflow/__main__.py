import click
from functools import wraps
from itertools import count, product
import numpy as np
from nutils import plot, log, function as fn, _
import pickle

import bbflow.cases as cases
import bbflow.quadrature as quadrature
import bbflow.solvers as solvers


def parse_extra_args(func):
    @wraps(func)
    def inner(ctx, **kwargs):
        extra_args = {}
        args = ctx.args
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
                if key.startswith('no-'):
                    extra_args[key[3:]] = False
                else:
                    extra_args[key] = True
            elif len(values) == 1:
                extra_args[key] = values[0]
            else:
                extra_args[key] = values
        return func(ctx, **kwargs, **extra_args)
    return inner


@click.group()
@click.pass_context
@click.option('--case', '-c', type=click.Choice(cases.__dict__), required=True)
@click.option('--solver', '-s', type=click.Choice(solvers.__all__), required=True)
def main(ctx, case, solver):
    ctx.obj = {
        'case': getattr(cases, case),
        'solver': getattr(solvers, solver),
    }

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
@parse_extra_args
def single(ctx, **kwargs):
    case = ctx.obj['case'](**kwargs)
    lhs = ctx.obj['solver'](case, **kwargs)
    solvers.metrics(case, lhs, **kwargs)
    solvers.plots(case, lhs, **kwargs)


@command('make-ensemble')
@click.option('--method', type=click.Choice(['pod']), default='pod')
@click.option('--imethod', type=click.Choice(['full', 'sparse']), default='full')
@parse_extra_args
@log.title
def make_ensemble(ctx, method, imethod, ipts=None, error=0.01, **kwargs):
    case = ctx.obj['case'](**kwargs)

    scheme = list(getattr(quadrature, imethod)(case.mu, ipts))
    nsnapshots = len(scheme)
    log.info('Generating ensemble of {} snapshots'.format(nsnapshots))

    ensemble = []
    for mu, weight in log.iter('snapshot', scheme):
        ensemble.append(weight * ctx.obj['solver'](case, mu=mu, **kwargs))
    ensemble = np.array(ensemble).T

    for field in case.fields:
        mass = case.mass(field)
        corr = ensemble.T.dot(mass.dot(ensemble))

        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:,::-1]

        threshold = (1 - error ** 2) * sum(eigvals)
        try:
            nmodes = min(np.where(np.cumsum(eigvals) > threshold)[0]) + 1
        except ValueError:
            nmodes = nsnapshots

        actual_error = np.sqrt(np.sum(eigvals[nmodes:]) / sum(eigvals))
        log.info('{} modes suffice for {:.2e} error (threshold {:.2e})'.format(
            nmodes, actual_error, error,
        ))

        if nmodes == nsnapshots:
            log.warning('All DoFs used, ensemble is probably too small')

        reduced = ensemble.dot(eigvecs[:,:nmodes]) / np.sqrt(eigvals[:nmodes])
        indices = case.basis_indices(field)
        mask = np.ones(reduced.shape[0], dtype=np.bool)
        mask[indices] = 0
        reduced[mask] = 0


if __name__ == '__main__':
    main()
