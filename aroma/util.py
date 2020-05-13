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
from numpy import newaxis as _
import h5py
import pyfive
from os.path import exists
import dill
import random
import scipy.sparse as sp
import scipy.sparse._sparsetools as sptools
import sharedmem
import string
import warnings
from io import BytesIO

try:
    import lrspline as lr
    has_lrspline = True
except ImportError:
    has_lrspline = False

from nutils import log, function as fn, topology, config


_SCALARS = (
    float, np.float, np.float128, np.float64, np.float32, np.float16,
    int, np.int, np.int64, np.int32, np.int16, np.int8, np.int0,
)


class FileBackedProperty:

    def __init__(self, attrname):
        self.attrname = attrname

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        if self.attrname in obj._file_data:
            return obj._file_data[self.attrname]
        if obj._group is None or self.attrname not in obj._group:
            raise AttributeError("Attribute {} not set".format(self.attrname))
        val = from_dataset(obj._group[self.attrname])
        obj._file_data[self.attrname] = val
        return val

    def __set__(self, obj, val):
        obj._file_data[self.attrname] = val

    def __delete__(self, obj):
        del obj._file_data[self.attrname]


class FileBackedMeta(type):

    def __new__(cls, name, bases, attrs):
        file_attribs = set(attrs.get('_file_attribs_', []))
        for base in bases:
            file_attribs |= set(getattr(base, '_file_attribs_', []))
        file_attribs = sorted(file_attribs)
        attrs['_file_attribs_'] = file_attribs
        for attr in file_attribs:
            attrs[attr] = FileBackedProperty(attr)
        return super().__new__(cls, name, bases, attrs)


class FileBacked(metaclass=FileBackedMeta):

    def __init__(self, group=None):
        self._file_data = {}
        self._group = group

    def write(self, group):
        group.attrs['module'] = self.__class__.__module__
        group.attrs['class'] = self.__class__.__name__
        for attr in self._file_attribs_:
            try:
                value = getattr(self, attr)
            except AttributeError:
                continue
            to_dataset(value, group, attr)

    @classmethod
    def read(cls, group):
        modulename = group.attrs['module']
        classname = group.attrs['class']
        for cls in subclasses(FileBacked, root=True):
            if cls.__module__ == modulename and cls.__name__ == classname:
                break
        else:
            raise TypeError(f"Unable to find appropriate class to load: {modulename}.{classname}")
        obj = cls.__new__(cls)
        FileBacked.__init__(obj, group)
        return obj


def to_dataset(obj, group, name):
    if isinstance(obj, FileBacked):
        subgroup = group.require_group(name)
        subgroup.attrs['type'] = np.string_('FileBacked')
        obj.write(subgroup)
        return subgroup

    if isinstance(obj, (sp.csr_matrix, sp.csc_matrix)):
        subgroup = group.require_group(name)
        subgroup['data'] = obj.data
        subgroup['indices'] = obj.indices
        subgroup['indptr'] = obj.indptr
        subgroup.attrs['shape'] = obj.shape
        subgroup.attrs['type'] = np.string_('CSRMatrix' if isinstance(obj, sp.csr_matrix) else 'CSCMatrix')
        return subgroup

    if isinstance(obj, sp.coo_matrix):
        subgroup = group.require_group(name)
        subgroup['data'] = obj.data
        subgroup['row'] = obj.row
        subgroup['col'] = obj.col
        subgroup.attrs['shape'] = obj.shape
        subgroup.attrs['type'] = np.string_('COOMatrix')

    if isinstance(obj, np.ndarray):
        group[name] = obj
        group[name].attrs['type'] = np.string_('Array')
        return group[name]

    if isinstance(obj, str):
        group[name] = np.string_(obj)
        group[name].attrs['type'] = np.string_('String')
        return group[name]

    if has_lrspline and isinstance(obj, lr.LRSplineSurface):
        with BytesIO() as b:
            obj.write(b)
            b.seek(0)
            group[name] = b.read()
        group[name].attrs['type'] = np.string_('LRSplineSurface')
        return group[name]

    else:
        group[name] = np.string_(dill.dumps(obj))
        group[name].attrs['type'] = np.string_('PickledObject')
        return group[name]


def from_dataset(group):
    type_ = group.attrs['type'].decode()
    if type_ == 'FileBacked':
        return FileBacked.read(group)
    if type_ == 'PickledObject':
        return dill.loads(group[()])
    if has_lrspline and type_ == 'LRSplineSurface':
        return lr.LRSplineSurface(group[()])
    if type_ == 'Array':
        return group[:]
    if type_ == 'String':
        return group[()].decode()
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
    name = name.decode('utf-8')
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
        with config(verbose=verbose, nprocs=nprocs), log.TeeLog(log.DataLog('.'), log.RichOutputLog()):
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
                    with pyfive.File(filename) as f:
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

    # Nutils will usually warn that explicit inflation is a bug
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = np.array([equation.eval(**kwg)[0] for kwg in kwargs])

    if equation.ndim == 2:
        data = np.transpose(data, (0, 2, 1))
        data = np.reshape(data, (ncomps * len(points), data.shape[-1]))
        data = sp.coo_matrix(data)
        data = sp.csr_matrix((data.data, (data.row + index, data.col)), shape=(size,)*2)
    elif equation.ndim == 1:
        data = np.hstack([np.zeros((index,)), data.flatten()])
    else:
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


class CSRAssembler:

    def __init__(self, shape, row, col):
        assert len(shape) == 2
        assert np.max(row) < shape[0]
        assert np.max(col) < shape[1]

        order = np.lexsort((row, col))
        row, col = row[order], col[order]
        mask = ((row[1:] != row[:-1]) | (col[1:] != col[:-1]))
        mask = np.append(True, mask)
        row, col = row[mask], col[mask]
        inds, = np.nonzero(mask)

        M, N = shape
        idx_dtype = sp.sputils.get_index_dtype((row, col), maxval=max(len(row), N))
        self.row = row.astype(idx_dtype, copy=False)
        self.col = col.astype(idx_dtype, copy=False)

        self.order, self.inds = order, inds
        self.shape = shape

    def write(self, group):
        to_dataset(self.row, group, 'row')
        to_dataset(self.col, group, 'col')
        to_dataset(self.order, group, 'order')
        to_dataset(self.inds, group, 'inds')
        group.attrs['shape'] = self.shape
        group.attrs['type'] = 'CSRAssembler'

    @staticmethod
    def read(group):
        retval = CSRAssembler.__new__(CSRAssembler)
        retval.row = group['row'][:]
        retval.col = group['col'][:]
        retval.order = group['order'][:]
        retval.inds = group['inds'][:]
        retval.shape = tuple(group.attrs['shape'])
        return retval

    def __call__(self, data):
        data = np.add.reduceat(data[self.order], self.inds)

        M, N = self.shape
        indptr = np.empty(M+1, dtype=self.row.dtype)
        indices = np.empty_like(self.col, dtype=self.row.dtype)
        new_data = np.empty_like(data)

        sptools.coo_tocsr(M, N, len(self.row), self.row, self.col, data, indptr, indices, new_data)
        return sp.csr_matrix((new_data, indices, indptr), shape=self.shape)

    def ensure_shareable(self):
        self.row, self.col, self.order, self.inds = map(
            shared_array, (self.row, self.col, self.order, self.inds)
        )


class VectorAssembler:

    def __init__(self, shape, inds):
        assert len(shape) == 1
        assert np.max(inds) < shape[0]

        order = np.lexsort((inds,))
        inds = inds[order]
        mask = inds[1:] != inds[:-1]
        mask = np.append(True, mask)
        self.row = inds[mask]
        inds, = np.nonzero(mask)

        self.order, self.inds = order, inds
        self.shape = shape

    def write(self, group):
        to_dataset(self.row, group, 'row')
        to_dataset(self.order, group, 'order')
        to_dataset(self.inds, group, 'inds')
        group.attrs['shape'] = self.shape
        group.attrs['type'] = 'VectorAssembler'

    @staticmethod
    def read(group):
        retval = VectorAssembler.__new__(VectorAssembler)
        retval.row = group['row'][:]
        retval.order = group['order'][:]
        retval.inds = group['inds'][:]
        retval.shape = tuple(group.attrs['shape'])
        return retval

    def __call__(self, data):
        data = np.add.reduceat(data[self.order], self.inds)
        retval = np.zeros(self.shape, dtype=data.dtype)
        retval[self.row] = data
        return retval

    def ensure_shareable(self):
        self.row, self.order, self.inds = map(shared_array, (self.row, self.order, self.inds))


def contract(obj, contraction):
    axes = []
    for i, cont in enumerate(contraction):
        if cont is None:
            continue
        assert cont.ndim == 1
        for __ in range(i):
            cont = cont[_,...]
        while cont.ndim < obj.ndim:
            cont = cont[...,_]
        obj = obj * cont
        axes.append(i)
    return obj.sum(tuple(axes))


def contract_sparse(obj, contraction):
    a, b = contraction
    if a is None and b is None:
        return obj
    if a is None and b is not None:
        return obj.dot(b)
    if a is not None and b is None:
        return obj.T.dot(a)
    return a.dot(obj.dot(b))
