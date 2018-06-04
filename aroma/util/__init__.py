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


import click
import inspect
import functools
import time as timemod
from multiprocessing import current_process, cpu_count
import numpy as np
import h5py
from os.path import exists
import pickle
import random
import scipy.sparse as sp
import sharedmem
import string

from nutils import log, function as fn, topology, config

from .sparse import SparseArray


_SCALARS = (
    float, np.float, np.float128, np.float64, np.float32, np.float16,
    int, np.int, np.int64, np.int32, np.int16, np.int8, np.int0,
)


def to_dataset(obj, group, name):
    if isinstance(obj, (fn.Array, topology.Topology, dict, type(None))):
        group[name] = np.string_(pickle.dumps(obj))
        group[name].attrs['type'] = 'PickledObject'
        return group[name]

    if isinstance(obj, (sp.csr_matrix, sp.csc_matrix)):
        subgroup = group.require_group(name)
        subgroup['data'] = obj.data
        subgroup['indices'] = obj.indices
        subgroup['indptr'] = obj.indptr
        subgroup.attrs['shape'] = obj.shape
        subgroup.attrs['type'] = 'CSRMatrix' if isinstance(obj, sp.csr_matrix) else 'CSCMatrix'
        return subgroup

    if isinstance(obj, sp.coo_matrix):
        subgroup = group.require_group(name)
        subgroup['data'] = obj.data
        subgroup['row'] = obj.row
        subgroup['col'] = obj.col
        subgroup.attrs['shape'] = obj.shape
        subgroup.attrs['type'] = 'COOMatrix'

    if isinstance(obj, np.ndarray):
        group[name] = obj
        group[name].attrs['type'] = 'Array'
        return group[name]

    if isinstance(obj, str):
        group[name] = obj
        group[name].attrs['type'] = 'String'
        return group[name]

    raise NotImplementedError(f'{type(obj)} to dataset')


def from_dataset(group):
    type_ = group.attrs['type']
    if type_ == 'PickledObject':
        return pickle.loads(group[()])
    if type_ == 'Array':
        return group[:]
    if type_ == 'String':
        return group[()]
    if type_ in {'CSRMatrix', 'CSCMatrix'}:
        cls = sp.csr_matrix if type_ == 'CSRMatrix' else sp.csc_matrix
        return cls((group['data'][:], group['indices'][:], group['indptr'][:]), shape=group.attrs['shape'])
    if type_ == 'COOMatrix':
        return sp.coo_matrix((group['data'][:], (group['row'][:], group['col'][:])), shape=group.attrs['shape'])

    raise NotImplementedError(f'Unknown type: {type_}')


def subclasses(cls, root=False):
    if root:
        yield cls
    for sub in cls.__subclasses__():
        yield sub
        yield from subclasses(sub, root=False)


def find_subclass(cls, name, root=False, attr='__name__'):
    for sub in subclasses(cls, root=root):
        if hasattr(sub, attr) and getattr(sub, attr) == name:
            return sub
    assert False


def make_filename(func, fmt, *args, **kwargs):
    signature = inspect.signature(func)
    arguments = [arg for __, arg, __, __ in string.Formatter().parse(fmt) if arg is not None]
    binding = signature.bind(*args, **kwargs)
    format_dict = {}
    for argname in arguments:
        if signature.parameters[argname].annotation is bool:
            prefix = '' if binding.arguments[argname] else 'no-'
            format_dict[argname] = f'{prefix}{argname}'
        else:
            format_dict[argname] = binding.arguments[argname]
    return fmt.format(**format_dict)


def common_args(func):
    @functools.wraps(func)
    def retval(verbose, nprocs, **kwargs):
        with config(verbose=verbose, nprocs=nprocs), log.RichOutputLog():
            return func(**kwargs)
    retval = click.option('--verbose', default=3)(retval)
    retval = click.option('--nprocs', default=cpu_count())(retval)
    return retval


def filecache(fmt):
    def decorator(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            from aroma.case import Case
            from aroma.ensemble import Ensemble
            filename = make_filename(func, fmt, *args, **kwargs)

            # If file exists, load from it
            with log.context(func.__name__):
                if exists(filename):
                    log.user(f'reading from {filename}')
                    reader = Case if filename.endswith('case') else Ensemble
                    with h5py.File(filename, 'r') as f:
                        return reader.read(f)
                log.user(f'{filename} not found')

            # If it doesn't exist, call the wrapped function, and save
            obj = func(*args, **kwargs)
            with log.context(func.__name__):
                log.user(f'writing to {filename}')
                with h5py.File(filename, 'w') as f:
                    obj.write(f)
            return obj

        return inner
    return decorator


def multiple_to_single(argname):
    def decorator(func):
        signature = inspect.signature(func)

        @functools.wraps(func)
        def ret(*args, **kwargs):
            binding = signature.bind(*args, **kwargs)
            args = binding.arguments[argname]
            multiple = True
            if not isinstance(args, (list, tuple)):
                multiple = False
                args = [args]
            retvals = []
            for arg in args:
                binding.arguments[argname] = arg
                retvals.append(func(*binding.args, **binding.kwargs))
            if not multiple:
                return retvals[0]
            return retvals

        return ret
    return decorator


class time:

    def __init__(self, context=None, display=True):
        self._display = True
        self._context = context
        self._time = 0.0

    def __enter__(self):
        self._time = timemod.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._time = timemod.time() - self._time
        if self._display:
            s = (str(self._context) + ': ') if self._context is not None else ''
            log.user('{}{:.2e} seconds'.format(s, self._time))

    @property
    def seconds(self):
        return self._time


def parallel_log(verbose=3, return_time=False):
    def decorator(func):
        @functools.wraps(func)
        def inner(args):
            count, *args = args
            with log.context(f'{current_process().name}'), log.context(f'{count}'), time() as t:
                retval = func(*args)
            if return_time:
                return t.seconds, retval
            return retval
        return inner
    return decorator


def collocate(domain, equation, points, index, size):
    ncomps = equation.shape[-1]

    elements = [domain.elements[eid] for eid, __ in points]
    kwargs = [{
        '_transforms': (elem.transform, elem.opposite),
        '_points': np.array([pt]),
    } for elem, (__, pt) in zip(elements, points)]

    data = np.array([equation.eval(**kwg)[0] for kwg in kwargs])

    if equation.ndim == 2:
        data = np.transpose(data, (0, 2, 1))
        data = np.reshape(data, (ncomps * len(points), data.shape[-1]))
        data = sp.coo_matrix(data)
        return SparseArray(data.data, (data.row + index, data.col), (size, size))
    elif equation.ndim == 1:
        return np.hstack([np.zeros((index,)), data.flatten()])
    raise NotImplementedError

    return data


def characteristic(domain, patches):
    basis = domain.basis_patch()
    return basis.dot([1 if i in patches else 0 for i in range(len(basis))])


def assembler(shape, *args):
    assert len(shape) == len(args)
    assert len(shape) <= 2

    backend = VectorAssembler if len(shape) == 1 else CSRAssembler
    return backend(shape, *args)


def shared_array(arr):
    ret = sharedmem.empty_like(arr)
    ret[:] = arr
    return ret
