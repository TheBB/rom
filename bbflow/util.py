import inspect
import functools
import time as timemod
from multiprocessing import current_process
import numpy as np
from os.path import exists
import pickle
import scipy.sparse as sp
import scipy.sparse._sparsetools as sptools
from nutils import log


def pickle_cache(fmt):
    def decorator(func):
        signature = inspect.signature(func)
        @functools.wraps(func)
        def inner(*args, **kwargs):
            binding = signature.bind(*args, **kwargs)
            filename = fmt.format(**binding.arguments)
            if exists(filename):
                with open(filename, 'rb') as f:
                    return pickle.load(f)
            obj = func(*args, **kwargs)
            with open(filename, 'wb') as f:
                pickle.dump(obj, f)
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


def parallel_log(verbose=3):
    def decorator(func):
        @functools.wraps(func)
        def inner(args):
            __verbose__ = verbose
            count, *args = args
            with log.context(f'{current_process().name}'), log.context(f'{count}'), time():
                return func(*args)
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

    def __call__(self, data):
        data = np.add.reduceat(data[self.order], self.inds)

        M, N = self.shape
        indptr = np.empty(M+1, dtype=self.idx_dtype)
        indices = np.empty_like(self.col, dtype=self.idx_dtype)
        new_data = np.empty_like(data)

        sptools.coo_tocsr(M, N, len(self.row), self.row, self.col, data, indptr, indices, new_data)
        return sp.csr_matrix((new_data, indices, indptr), shape=self.shape)


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

    def __call__(self, data):
        data = np.add.reduceat(data[self.order], self.inds)
        retval = np.zeros(self.shape, dtype=data.dtype)
        retval[self.row] = data
        return retval
