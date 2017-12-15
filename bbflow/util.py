import inspect
import functools
import time as timemod
from multiprocessing import current_process
import numpy as np
import h5py
from os.path import exists
import pickle
import random
import scipy.sparse as sp
import scipy.sparse._sparsetools as sptools
import sharedmem
import string
from nutils import log


_h5file = None


def h5pickle():
    return _h5file is not None


def dump_array(obj):
    assert h5pickle()
    key = ''.join(random.choices('0123456789abcdef', k=16))
    while key in _h5file:
        key = ''.join(random.choices('0123456789abcdef', k=16))
    _h5file.create_dataset(key, data=obj)
    return key


def load_array(key):
    assert h5pickle()
    assert key in _h5file
    return np.array(_h5file[key])


def dump(obj, filename):
    global _h5file
    _h5file = h5py.File(filename, 'w')

    bytedata = np.fromstring(pickle.dumps(obj), dtype=np.uint8)
    _h5file.create_dataset('bytedata', data=bytedata)

    _h5file.close()
    _h5file = None


def load(filename):
    global _h5file
    _h5file = h5py.File(filename, 'r')

    bytedata = np.array(_h5file['bytedata']).tostring()
    obj = pickle.loads(bytedata)

    _h5file.close()
    _h5file = None

    return obj


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


def pickle_cache(fmt):
    def decorator(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            filename = make_filename(func, fmt, *args, **kwargs)
            with log.context(func.__name__):
                if exists(filename):
                    log.user(f'reading from {filename}')
                    return load(filename)
                log.user(f'{filename} not found')
            obj = func(*args, **kwargs)
            with log.context(func.__name__):
                log.user(f'writing to {filename}')
            dump(obj, filename)
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
            __verbose__ = verbose
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
        self.idx_dtype = idx_dtype

    def __getstate__(self):
        if not h5pickle():
            return self.__dict__
        return {
            'shape': self.shape,
            'idx_dtype': self.idx_dtype,
            'row': dump_array(self.row),
            'col': dump_array(self.col),
            'order': dump_array(self.order),
            'inds': dump_array(self.inds)
        }

    def __setstate__(self, state):
        if not h5pickle():
            self.__dict__.update(state)
            return
        self.shape = state['shape']
        self.idx_dtype = state['idx_dtype']
        self.row = load_array(state['row'])
        self.col = load_array(state['col'])
        self.order = load_array(state['order'])
        self.inds = load_array(state['inds'])

    def __call__(self, data):
        data = np.add.reduceat(data[self.order], self.inds)

        M, N = self.shape
        indptr = np.empty(M+1, dtype=self.idx_dtype)
        indices = np.empty_like(self.col, dtype=self.idx_dtype)
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

    def __getstate__(self):
        if not h5pickle():
            return self.__dict__
        return {
            'shape': self.shape,
            'row': dump_array(self.row),
            'order': dump_array(self.order),
            'inds': dump_array(self.inds)
        }

    def __setstate__(self, state):
        if not h5pickle():
            self.__dict__.update(state)
            return
        self.shape = state['shape']
        self.row = load_array(state['row'])
        self.order = load_array(state['order'])
        self.inds = load_array(state['inds'])

    def __call__(self, data):
        data = np.add.reduceat(data[self.order], self.inds)
        retval = np.zeros(self.shape, dtype=data.dtype)
        retval[self.row] = data
        return retval

    def ensure_shareable(self):
        self.row, self.order, self.inds = map(shared_array, (self.row, self.order, self.inds))
